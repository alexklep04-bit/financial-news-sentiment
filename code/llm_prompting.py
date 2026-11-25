# Decoder-only LLM (Qwen2.5-Instruct) used without any training: zero-shot and few-shot prompting.
# Instead of letting the model generate free text (and then parsing it), we look at the
# probability of the next token after the prompt and compare "positive" / "negative" / "neutral".
# This way every answer is a valid label and we also get class probabilities.
#
# usage: python code/llm_prompting.py Qwen/Qwen2.5-1.5B-Instruct zero
#        python code/llm_prompting.py Qwen/Qwen2.5-1.5B-Instruct few 1 2 3 4 5
#        add --pb-only to skip the tweets (they take ~70% of the time)
import copy
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import LABELS, load_split, save_preds, get_device

device = get_device()

INSTRUCTION = (
    "You are a financial analyst. Read the sentence from a financial news text and decide how it "
    "would most likely affect the stock price of the company it talks about, from the point of view "
    "of an investor. Answer with exactly one word: positive, negative or neutral.\n\nSentence: {text}"
)
K_PER_CLASS = 2      # few-shot: 2 examples of every class = 6 demonstrations


def build_messages(text, demos, template=INSTRUCTION):
    msgs = []
    for d_text, d_label in demos:
        msgs.append({"role": "user", "content": template.format(text=d_text)})
        msgs.append({"role": "assistant", "content": d_label})
    msgs.append({"role": "user", "content": template.format(text=text)})
    return msgs


def label_token_ids(tok):
    # several spellings can start the answer, we add up their probabilities
    ids = {}
    for lab in LABELS:
        variants = {lab, lab.capitalize(), " " + lab, " " + lab.capitalize()}
        ids[lab] = sorted({tok.encode(v, add_special_tokens=False)[0] for v in variants})
    # the first tokens must not overlap between classes
    flat = [i for v in ids.values() for i in v]
    assert len(flat) == len(set(flat)), ids
    return ids


@torch.no_grad()
def classify(model, tok, texts, demos, bs=16, template=INSTRUCTION):
    lab_ids = label_token_ids(tok)
    prompts = [tok.apply_chat_template(build_messages(t, demos, template), tokenize=False, add_generation_prompt=True)
               for t in texts]
    # Every prompt starts with the same system message, demonstrations and instruction, only the
    # sentence at the end is different. We run this common part through the model once and reuse
    # its key/value cache, otherwise 6-shot is several times slower than 0-shot.
    # (we cut right after the last ":" of "Sentence:" / "Headline:", so the tokens stay the same)
    common = os.path.commonprefix(prompts)
    cut = common.rfind(":") + 1
    prefix = prompts[0][:cut]
    assert all(p.startswith(prefix) for p in prompts)
    pre_ids = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    pre_cache = model(pre_ids, use_cache=True).past_key_values

    suffixes = [p[cut:] for p in prompts]
    tok.padding_side = "right"     # padding at the end, so the cached prefix stays in front of every row
    order = np.argsort([len(s) for s in suffixes])   # similar lengths in one batch -> less padding
    probs = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        enc = tok([suffixes[k] for k in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        cache = copy.deepcopy(pre_cache)
        cache.batch_repeat_interleave(len(idx))
        ones = torch.ones(len(idx), pre_ids.shape[1], dtype=enc.attention_mask.dtype, device=device)
        out = model(input_ids=enc.input_ids, attention_mask=torch.cat([ones, enc.attention_mask], 1),
                    past_key_values=cache)
        last = enc.attention_mask.sum(1) - 1        # last real token of each row
        p = torch.softmax(out.logits[torch.arange(len(idx)), last].float(), -1)
        probs[idx] = torch.stack([p[:, lab_ids[lab]].sum(1) for lab in LABELS], 1).cpu().numpy()
    mass = probs.sum(1)          # how much probability went to one of the label words at all
    return probs / mass[:, None], mass


def sample_demos(train, seed):
    rng = np.random.RandomState(seed)
    demos = []
    for lab in LABELS:
        rows = train[train["label"] == lab]
        for j in rng.choice(len(rows), K_PER_CLASS, replace=False):
            demos.append((rows.iloc[j]["text"], lab))
    rng.shuffle(demos)
    return demos


# target-aware version for the case study: the question is about ONE firm, not the whole sentence
TARGET_INSTRUCTION = (
    "You are a financial analyst. Read the news headline and decide how it would most likely affect "
    "{firm} (its business or its stock price). Be careful: very often {firm} is only mentioned because "
    "its analysts rate some other company, or because it owns shares of another company, or the name "
    "is used for something else. In such cases the news is neutral for {firm}. "
    "Answer with exactly one word: positive, negative or neutral.\n\nHeadline: {text}"
)


def run_manual_sample(model, tok, short):
    ann = pd.read_csv("data/manual_sample_annotated.csv")
    out = ann[["headline", "firm"]].copy()
    probs, _ = classify(model, tok, list(ann["headline"]), [])
    out[f"pred_{short}_generic"] = [LABELS[i] for i in probs.argmax(1)]

    preds = {}
    for firm, idx in ann.groupby("firm").groups.items():
        template = TARGET_INSTRUCTION.replace("{firm}", firm)
        p, _ = classify(model, tok, list(ann.loc[idx, "headline"]), [], template=template)
        preds.update(zip(idx, [LABELS[i] for i in p.argmax(1)]))
    out[f"pred_{short}_target"] = pd.Series(preds)
    out.to_csv(f"results/manual_llm_{short}.csv", index=False)
    print(out.iloc[:, 2:].apply(lambda c: c.value_counts()).to_string())


if __name__ == "__main__":
    pb_only = "--pb-only" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--pb-only"]
    model_id, mode = args[0], args[1]
    seeds = [int(s) for s in args[2:]] or [0]
    short = model_id.split("/")[-1].replace("-Instruct", "").lower()       # e.g. qwen2.5-1.5b

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(device).eval()

    if mode == "target":
        run_manual_sample(model, tok, short)
        sys.exit()

    train, test = load_split("pb_train"), load_split("pb_test")
    tw = load_split("twitter_test")
    tw = tw[(tw["split"] == "valid") | tw["firm"].notna()].reset_index(drop=True)   # keeps it affordable

    for seed in seeds:
        demos = sample_demos(train, seed) if mode == "few" else []
        name = f"{short}_{mode}shot"
        for dname, d in [("pb_test", test)] + ([] if pb_only else [("twitter", tw)]):
            t0 = time.time()
            probs, mass = classify(model, tok, list(d["text"]), demos)
            print(f"  {name} seed {seed} {dname}: {len(d)} texts in {time.time() - t0:.0f}s, "
                  f"avg prob. mass on label words = {mass.mean():.3f}")
            save_preds(name, dname, seed, d, [LABELS[i] for i in probs.argmax(1)], probs)
