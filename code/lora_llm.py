# Parameter-efficient fine-tuning (LoRA) of the same decoder LLM that we use for prompting.
# The LM head is replaced by a 3-class classification head on the last token, the base weights
# stay frozen and only the low-rank adapters (+ the new head) are trained.
#
# usage: python code/lora_llm.py Qwen/Qwen2.5-1.5B-Instruct 1 2 3 4 5
import sys
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score

from utils import LABELS, LAB2ID, load_split, save_preds, set_seed, get_device

device = get_device()
EPOCHS = 3
LR = 2e-4
BATCH = 16
MAX_LEN = 96


def batches(tok, texts, labels=None, bs=BATCH, shuffle=False):
    idx = np.random.permutation(len(texts)) if shuffle else np.arange(len(texts))
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        enc = tok([texts[k] for k in j], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        if labels is not None:
            enc["labels"] = torch.tensor([labels[k] for k in j]).to(device)
        yield enc


@torch.no_grad()
def predict(model, tok, texts):
    model.eval()
    out = []
    for b in batches(tok, list(texts), bs=32):
        out.append(torch.softmax(model(**b).logits.float(), -1).cpu())
    return torch.cat(out).numpy()


def run(model_id, seed, train, val, tests):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "right"
    # float32: bf16 base weights + fp32 adapters crashed in backward on mps
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3, dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id       # needed to find the last real token

    cfg = LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, cfg)
    if seed == 1:
        model.print_trainable_parameters()
    model.to(device)

    X, y = list(train["text"]), list(train["label"].map(LAB2ID))
    steps = EPOCHS * int(np.ceil(len(X) / BATCH))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    best_f1, best_state = -1, None
    for ep in range(EPOCHS):
        model.train()
        t0, losses = time.time(), []
        for b in batches(tok, X, y, shuffle=True):
            loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
        vf1 = f1_score(val["label"].map(LAB2ID), predict(model, tok, val["text"]).argmax(1), average="macro")
        print(f"  lora seed {seed} epoch {ep + 1}: loss {np.mean(losses):.4f}  val F1 {vf1:.4f}  ({time.time() - t0:.0f}s)")
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

    # put the best adapter weights back
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in best_state:
                p.copy_(best_state[n].to(device))

    short = model_id.split("/")[-1].replace("-Instruct", "").lower()
    for dname, d in tests:
        probs = predict(model, tok, d["text"])
        save_preds(f"{short}_lora", dname, seed, d, [LABELS[i] for i in probs.argmax(1)], probs)


if __name__ == "__main__":
    model_id = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2:]] or [1, 2, 3, 4, 5]
    train, val, test = load_split("pb_train"), load_split("pb_val"), load_split("pb_test")
    tw = load_split("twitter_test")
    print("device:", device)
    for seed in seeds:
        run(model_id, seed, train, val, [("pb_test", test), ("twitter", tw)])
        if device.type == "mps":
            torch.mps.empty_cache()
