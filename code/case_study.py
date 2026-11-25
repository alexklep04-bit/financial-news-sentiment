# Case study: what does a sentence-level sentiment model say about JPMorgan, Goldman Sachs,
# BlackRock, Vanguard and Citadel?
#
#   python code/case_study.py sample    -> draws a random sample of headlines for manual annotation
#   python code/case_study.py analyse   -> uses FinBERT predictions (5 seeds, averaged) + the annotations
import glob
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from utils import LABELS

FIRMS = ["JPMorgan", "Goldman Sachs", "BlackRock", "Vanguard", "Citadel"]
PER_FIRM = 40


def make_sample():
    heads = pd.read_csv("data/firm_headlines.csv")
    parts = []
    for firm in FIRMS:
        g = heads[heads["firm"] == firm]
        parts.append(g.sample(min(PER_FIRM, len(g)), random_state=2024))
    s = pd.concat(parts).reset_index(drop=True)
    # annotation columns:
    #   about_firm    : 1 if the headline is news about the firm itself, 0 if the firm is only mentioned
    #                   (as a broker, a shareholder, a name of something else, ...)
    #   sent_sentence : overall sentiment of the headline (same guideline as PhraseBank)
    #   sent_firm     : sentiment for the firm's own stock / reputation
    for c in ["about_firm", "sent_sentence", "sent_firm"]:
        s[c] = ""
    s.to_csv("data/manual_sample_to_annotate.csv", index=False)
    print("sample saved:", len(s), s["firm"].value_counts().to_dict())


def ensemble(model, dataset, seeds=(1, 2, 3, 4, 5)):
    probs = None
    for seed in seeds:
        d = pd.read_csv(f"results/preds/{model}__{dataset}__seed{seed}.csv")
        p = d[[f"p_{l}" for l in LABELS]].values
        probs = p if probs is None else probs + p
    probs = probs / len(seeds)
    return d["text"].values, probs


def analyse():
    heads = pd.read_csv("data/firm_headlines.csv", parse_dates=["date"])
    texts, probs = ensemble("finbert", "firm_heads")
    assert (texts == heads["headline"].values).all()
    heads["pred"] = [LABELS[i] for i in probs.argmax(1)]
    heads["score"] = probs[:, 2] - probs[:, 0]       # P(pos) - P(neg), between -1 and 1
    heads.to_csv("results/firm_headlines_scored.csv", index=False)

    print("=== predicted sentiment of all filtered headlines per firm ===")
    tab = pd.crosstab(heads["firm"], heads["pred"], normalize="index").reindex(FIRMS)[LABELS]
    tab["n"] = heads["firm"].value_counts().reindex(FIRMS)
    tab["net"] = tab["positive"] - tab["negative"]
    print(tab.round(3).to_string())
    tab.to_csv("results/case_firm_shares.csv")

    print("\n=== net sentiment per year (share pos - share neg), firms with enough news ===")
    heads["year"] = heads["date"].dt.year
    yearly = (heads[heads["firm"].isin(FIRMS[:3])]
              .groupby(["firm", "year"])["pred"]
              .agg(n="size",
                   net=lambda s: (s == "positive").mean() - (s == "negative").mean()))
    print(yearly.unstack(0).round(2).to_string())
    yearly.reset_index().to_csv("results/case_yearly.csv", index=False)

    for firm in FIRMS[:3]:
        g = heads[heads["firm"] == firm].sort_values("score")
        print(f"\n{firm}: most negative ->", g["headline"].head(3).tolist())
        print(f"{firm}: most positive ->", g["headline"].tail(3).tolist())

    # ---- manual sample ----
    ann = pd.read_csv("data/manual_sample_annotated.csv")
    ann = ann.merge(heads[["headline", "firm", "pred", "score"]].rename(columns={"pred": "pred_finbert"}),
                    on=["headline", "firm"], how="left")
    # predictions of the prompted LLMs (generic prompt and the firm-aware prompt), see llm_prompting.py
    for f in sorted(glob.glob("results/manual_llm_*.csv")):
        ann = ann.merge(pd.read_csv(f), on=["headline", "firm"], how="left")
    assert ann["pred_finbert"].notna().all()
    print(f"\n=== manual sample: {len(ann)} headlines ===")
    print("share of headlines that are really about the firm:",
          ann.groupby("firm")["about_firm"].mean().reindex(FIRMS).round(2).to_dict(),
          "overall", round(ann["about_firm"].mean(), 3))
    print("gold sentence-level label distribution:", ann["sent_sentence"].value_counts().to_dict())
    print("label differs between sentence and firm level in",
          (ann["sent_sentence"] != ann["sent_firm"]).sum(), "headlines")

    res = []
    for model_col in [c for c in ann.columns if c.startswith("pred")]:
        for target in ["sent_sentence", "sent_firm"]:
            acc = accuracy_score(ann[target], ann[model_col])
            f1 = f1_score(ann[target], ann[model_col], average="macro", labels=LABELS)
            res.append((model_col, target, acc, f1))
            print(f"  {model_col:22s} vs {target:14s}: acc {acc:.3f}  macro-F1 {f1:.3f}")
    pd.DataFrame(res, columns=["model", "gold", "acc", "f1"]).to_csv("results/case_manual_eval.csv", index=False)
    ann.to_csv("results/manual_sample_with_preds.csv", index=False)


if __name__ == "__main__":
    {"sample": make_sample, "analyse": analyse}[sys.argv[1]]()
