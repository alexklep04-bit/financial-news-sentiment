# Reads every prediction file in results/preds and makes the result tables.
# For models trained with several seeds we report mean +- 95% confidence interval
#   CI = mean +- 1.96 * s / sqrt(n)      (s = sample standard deviation, n = number of runs)
import glob
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

from utils import LABELS, PRED_DIR

NICE = {  # order + names used in the report
    "majority": "Majority class",
    "lm_lexicon": "LM finance lexicon",
    "tfidf_lr": "TF-IDF + LogReg",
    "tfidf_lr_regex": "TF-IDF + LogReg (regex norm.)",
    "mlp_glove": "FFNN, avg. GloVe",
    "mlp_fin": "FFNN, avg. FinText",
    "cnn_glove": "CNN, GloVe",
    "cnn_fin": "CNN, FinText",
    "bilstm_glove": "BiLSTM, GloVe",
    "bilstm_fin": "BiLSTM, FinText",
    "bert": "BERT-base",
    "finbert": "FinBERT",
    "qwen2.5-0.5b_zeroshot": "Qwen2.5-0.5B zero-shot",
    "qwen2.5-1.5b_zeroshot": "Qwen2.5-1.5B zero-shot",
    "qwen2.5-3b_zeroshot": "Qwen2.5-3B zero-shot",
    "qwen2.5-1.5b_fewshot": "Qwen2.5-1.5B 6-shot",
    "qwen2.5-3b_fewshot": "Qwen2.5-3B 6-shot",
}


def ci(values):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return v.mean(), 0.0
    return v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def load_all():
    rows = []
    for f in glob.glob(f"{PRED_DIR}/*.csv"):
        model, dataset, seed = os.path.basename(f)[:-4].split("__")
        rows.append((model, dataset, int(seed.replace("seed", "")), f))
    return pd.DataFrame(rows, columns=["model", "dataset", "seed", "file"])


def metrics(df):
    acc = accuracy_score(df["gold"], df["pred"])
    f1 = f1_score(df["gold"], df["pred"], average="macro", labels=LABELS)
    per_class = f1_score(df["gold"], df["pred"], average=None, labels=LABELS)
    return acc, f1, per_class


def summarise(files, subset=None):
    """subset: function(df) -> boolean mask, to evaluate only on a part of the test set"""
    out = []
    for model, g in files.groupby("model"):
        accs, f1s, pcs = [], [], []
        for f in g["file"]:
            df = pd.read_csv(f)
            if subset is not None:
                df = df[subset(df)]
            a, m, pc = metrics(df)
            accs.append(a); f1s.append(m); pcs.append(pc)
        acc_m, acc_ci = ci(accs)
        f1_m, f1_ci = ci(f1s)
        pcs = np.mean(pcs, axis=0)
        out.append(dict(model=model, name=NICE.get(model, model), runs=len(g), n=len(df),
                        acc=acc_m, acc_ci=acc_ci, f1=f1_m, f1_ci=f1_ci,
                        f1_neg=pcs[0], f1_neu=pcs[1], f1_pos=pcs[2]))
    out = pd.DataFrame(out)
    out["order"] = out["model"].map({k: i for i, k in enumerate(NICE)}).fillna(99)
    return out.sort_values("order").drop(columns="order").reset_index(drop=True)


def fmt(m, c):
    return f"{100 * m:.1f} ± {100 * c:.1f}" if c > 0 else f"{100 * m:.1f}"


if __name__ == "__main__":
    files = load_all()
    pb_test = pd.read_csv("data/pb_test.csv")
    tw = pd.read_csv("data/twitter_test.csv")
    agreement = dict(zip(pb_test["text"], pb_test["agreement"]))
    tw_split = dict(zip(tw["text"], tw["split"]))
    tw_firm = dict(zip(tw["text"], tw["firm"].fillna("")))

    # ---- main table, PhraseBank test ----
    pb = summarise(files[files["dataset"] == "pb_test"])
    pb.to_csv("results/summary_phrasebank.csv", index=False)
    print("\n=== Financial PhraseBank, test (968 sentences) ===")
    for _, r in pb.iterrows():
        print(f"{r['name']:32s} runs={r['runs']}  acc {fmt(r['acc'], r['acc_ci']):>12s}   macro-F1 {fmt(r['f1'], r['f1_ci']):>12s}"
              f"   F1 neg/neu/pos {100*r['f1_neg']:.1f}/{100*r['f1_neu']:.1f}/{100*r['f1_pos']:.1f}")

    # ---- by annotator agreement ----
    rows = []
    for level in ["100%", "75-99%", "66-74%", "50-65%"]:
        s = summarise(files[files["dataset"] == "pb_test"], lambda d, l=level: d["text"].map(agreement) == l)
        s["agreement"] = level
        rows.append(s)
    agr = pd.concat(rows)
    agr.to_csv("results/summary_agreement.csv", index=False)
    print("\n=== accuracy by agreement level ===")
    print(agr.pivot_table(index="name", columns="agreement", values="acc", sort=False)
          [["100%", "75-99%", "66-74%", "50-65%"]].mul(100).round(1).to_string())
    print("sentences per level:", pb_test["agreement"].value_counts().to_dict())

    # ---- out of domain: tweets ----
    tw_files = files[files["dataset"] == "twitter"]
    valid = summarise(tw_files, lambda d: d["text"].map(tw_split) == "valid")
    firms = summarise(tw_files, lambda d: d["text"].map(tw_firm) != "")
    valid.to_csv("results/summary_twitter_valid.csv", index=False)
    firms.to_csv("results/summary_twitter_firms.csv", index=False)
    print("\n=== Twitter financial news (out of domain) ===")
    m = valid.merge(firms[["model", "acc", "f1", "acc_ci", "f1_ci"]], on="model", suffixes=("", "_firm"))
    for _, r in m.iterrows():
        print(f"{r['name']:32s} valid: acc {fmt(r['acc'], r['acc_ci']):>12s} F1 {fmt(r['f1'], r['f1_ci']):>12s}   "
              f"| firm tweets: acc {fmt(r['acc_firm'], r['acc_ci_firm']):>12s} F1 {fmt(r['f1_firm'], r['f1_ci_firm']):>12s}")

    # ---- confusion matrices (summed over seeds) ----
    print("\n=== confusion matrices, PhraseBank test, rows = gold, cols = predicted (neg, neu, pos) ===")
    for model in ["tfidf_lr_regex", "cnn_glove", "bert", "finbert", "qwen2.5-1.5b_zeroshot", "qwen2.5-1.5b_fewshot", "qwen2.5-3b_fewshot"]:
        g = files[(files["model"] == model) & (files["dataset"] == "pb_test")]
        if len(g) == 0:
            continue
        cm = sum(confusion_matrix(pd.read_csv(f)["gold"], pd.read_csv(f)["pred"], labels=LABELS) for f in g["file"])
        print(model, "\n", cm)
