import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

LABELS = ["negative", "neutral", "positive"]
LAB2ID = {l: i for i, l in enumerate(LABELS)}
PRED_DIR = "results/preds"


def load_split(name):
    return pd.read_csv(f"data/{name}.csv")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def save_preds(model_name, dataset, seed, df, preds, probs=None):
    """one csv per (model, dataset, run) -> evaluate.py reads them all later"""
    os.makedirs(PRED_DIR, exist_ok=True)
    out = pd.DataFrame({"text": df["text"].values, "gold": df["label"].values, "pred": list(preds)})
    if probs is not None:
        probs = np.asarray(probs)
        for i, l in enumerate(LABELS):
            out[f"p_{l}"] = probs[:, i]
    out.to_csv(f"{PRED_DIR}/{model_name}__{dataset}__seed{seed}.csv", index=False)
    acc = accuracy_score(out["gold"], out["pred"])
    f1 = f1_score(out["gold"], out["pred"], average="macro", labels=LABELS)
    print(f"   {model_name:22s} {dataset:10s} seed={seed}  acc={acc:.4f}  macroF1={f1:.4f}")
    return acc, f1


def get_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
