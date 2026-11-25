# Fine-tuning of encoder transformers for 3-class sentiment
#   bert    - bert-base-uncased (general domain: Wikipedia + BooksCorpus)
#   finbert - yiyanghkust/finbert-pretrain (same BERT-base architecture, pre-trained on
#             ~4.9B tokens of 10-K/10-Q reports, earnings call transcripts and analyst reports)
# NB: we do NOT use ProsusAI/finbert, because it was already fine-tuned on Financial PhraseBank,
#     so its scores on our test split would be meaningless.
#
# usage: python code/finetune_bert.py bert   |   python code/finetune_bert.py finbert
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

from utils import LABELS, LAB2ID, load_split, save_preds, set_seed, get_device

MODELS = {"bert": "bert-base-uncased", "finbert": "yiyanghkust/finbert-pretrain"}
SEEDS = [1, 2, 3, 4, 5]
EPOCHS = 4
LR = 2e-5
BATCH = 16
MAX_LEN = 96

device = get_device()


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
    probs = []
    for b in batches(tok, list(texts), bs=64):
        probs.append(torch.softmax(model(**b).logits.float(), -1).cpu())
    return torch.cat(probs).numpy()


def run(key, seed, train, val, tests):
    set_seed(seed)
    # finbert-pretrain only ships a vocab.txt, AutoTokenizer can't figure it out -> load it as a BERT tokenizer
    tok = BertTokenizerFast.from_pretrained(MODELS[key], do_lower_case=True)
    # its config.json has no model_type either, so again we say explicitly that it is a BERT
    model = BertForSequenceClassification.from_pretrained(MODELS[key], num_labels=3).to(device)

    X, y = list(train["text"]), list(train["label"].map(LAB2ID))
    steps = EPOCHS * int(np.ceil(len(X) / BATCH))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    best_f1, best_state = -1, None
    for ep in range(EPOCHS):
        model.train()
        t0, losses = time.time(), []
        for b in batches(tok, X, y, shuffle=True):
            loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
        vf1 = f1_score(val["label"].map(LAB2ID), predict(model, tok, val["text"]).argmax(1), average="macro")
        print(f"  {key} seed {seed} epoch {ep + 1}: loss {np.mean(losses):.4f}  val F1 {vf1:.4f}  ({time.time() - t0:.0f}s)")
        if vf1 > best_f1:   # keep the best epoch (early stopping on validation)
            best_f1 = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    for dname, d in tests:
        probs = predict(model, tok, d["text"])
        save_preds(key, dname, seed, d, [LABELS[i] for i in probs.argmax(1)], probs)
    return model, tok


if __name__ == "__main__":
    key = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2:]] or SEEDS
    train, val, test = load_split("pb_train"), load_split("pb_val"), load_split("pb_test")
    tw = load_split("twitter_test")
    heads = load_split("firm_headlines").rename(columns={"headline": "text"})
    heads["label"] = "neutral"   # dummy, these headlines have no gold labels

    print("device:", device)
    for seed in seeds:
        model, tok = run(key, seed, train, val, [("pb_test", test), ("twitter", tw), ("firm_heads", heads)])
        if key == "finbert" and seed == 1:
            model.save_pretrained("results/finbert_seed1")
            tok.save_pretrained("results/finbert_seed1")
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
