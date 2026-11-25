# Figures for the report and the slides (saved to figures/ as png, 200 dpi)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from utils import LABELS

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b", "axes.titlesize": 10, "axes.titleweight": "bold", "figure.dpi": 200,
    "savefig.bbox": "tight", "legend.frameon": False,
})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"
GRAY = "#a3a29b"
SENT_COL = {"negative": RED, "neutral": "#c9c8c2", "positive": BLUE}

FAMILY = {
    "majority": ("Baseline", GRAY), "lm_lexicon": ("Lexicon / bag-of-words", BLUE),
    "tfidf_lr": ("Lexicon / bag-of-words", BLUE), "tfidf_lr_regex": ("Lexicon / bag-of-words", BLUE),
    "mlp_glove": ("Word embeddings + NN", ORANGE), "mlp_fin": ("Word embeddings + NN", ORANGE),
    "cnn_glove": ("Word embeddings + NN", ORANGE), "cnn_fin": ("Word embeddings + NN", ORANGE),
    "bilstm_glove": ("Word embeddings + NN", ORANGE), "bilstm_fin": ("Word embeddings + NN", ORANGE),
    "bert": ("Fine-tuned encoder (BERT)", AQUA), "finbert": ("Fine-tuned encoder (BERT)", AQUA),
}


def family(model):
    return FAMILY.get(model, ("Decoder LLM (Qwen2.5)", VIOLET))


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_main_results():
    pb = pd.read_csv("results/summary_phrasebank.csv")
    pb = pb.iloc[::-1]   # barh draws from the bottom, so reverse to keep the table order
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    y = np.arange(len(pb))
    cols = [family(m)[1] for m in pb["model"]]
    ax.barh(y, pb["f1"] * 100, height=0.62, color=cols, xerr=pb["f1_ci"] * 100,
            error_kw=dict(ecolor=INK2, elinewidth=0.9, capsize=2))
    for yi, v, c in zip(y, pb["f1"] * 100, pb["f1_ci"] * 100):
        ax.text(v + c + 0.8, yi, f"{v:.1f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, pb["name"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Macro-F1 on the PhraseBank test set, % (error bars: 95% CI over 5 runs)")
    grid(ax)
    seen = {}
    for m in pb["model"][::-1]:
        f, c = family(m)
        seen.setdefault(f, c)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in seen.values()]
    ax.legend(handles, seen.keys(), loc="upper center", bbox_to_anchor=(0.4, -0.11), ncol=3, fontsize=7.5)
    fig.savefig("figures/fig_main_results.png")
    plt.close(fig)


def fig_agreement():
    agr = pd.read_csv("results/summary_agreement.csv")
    keep = {"tfidf_lr_regex": BLUE, "cnn_glove": ORANGE, "bert": AQUA, "finbert": YELLOW,
            "qwen2.5-1.5b_fewshot": VIOLET}
    levels = ["100%", "75-99%", "66-74%", "50-65%"]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for m, c in keep.items():
        d = agr[agr["model"] == m].set_index("agreement").reindex(levels)
        if d["acc"].isna().all():
            continue
        ax.plot(range(4), d["acc"] * 100, color=c, lw=2, marker="o", ms=5,
                markeredgecolor="white", markeredgewidth=1.2, label=d["name"].iloc[0])
    maj = agr[agr["model"] == "majority"].set_index("agreement").reindex(levels)
    ax.plot(range(4), maj["acc"] * 100, color=GRAY, lw=1.5, ls="--", label="Majority class")
    ax.set_xticks(range(4), [f"{l}\n(n={n})" for l, n in zip(levels, agr[agr["model"] == "finbert"]
                                                               .set_index("agreement").reindex(levels)["n"])])
    ax.set_xlabel("Share of annotators that agreed on the gold label")
    ax.set_ylabel("Accuracy, %")
    ax.set_ylim(30, 100)
    grid(ax, "y")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    fig.savefig("figures/fig_agreement.png")
    plt.close(fig)


def fig_domain_shift():
    pb = pd.read_csv("results/summary_phrasebank.csv").set_index("model")
    tv = pd.read_csv("results/summary_twitter_valid.csv").set_index("model")
    tf = pd.read_csv("results/summary_twitter_firms.csv").set_index("model")
    models = [m for m in ["lm_lexicon", "tfidf_lr_regex", "cnn_glove", "cnn_fin", "bert", "finbert",
                          "qwen2.5-1.5b_zeroshot", "qwen2.5-1.5b_fewshot", "qwen2.5-3b_zeroshot"]
              if m in tv.index]
    short = {"lm_lexicon": "LM\nlexicon", "tfidf_lr_regex": "TF-IDF\n+ LogReg", "cnn_glove": "CNN\nGloVe",
             "cnn_fin": "CNN\nFinText", "bert": "BERT", "finbert": "FinBERT", "qwen2.5-1.5b_zeroshot": "Qwen\n1.5B\n0-shot",
             "qwen2.5-1.5b_fewshot": "Qwen\n1.5B\n6-shot", "qwen2.5-3b_zeroshot": "Qwen\n3B\n0-shot"}
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    x = np.arange(len(models))
    w = 0.27
    for k, (df, lab, c) in enumerate([(pb, "PhraseBank test (in-domain)", BLUE),
                                      (tv, "Tweets, valid split", ORANGE),
                                      (tf, "Tweets about the 5 firms (n=125)", AQUA)]):
        ax.bar(x + (k - 1) * w, df.loc[models, "f1"] * 100, width=w - 0.03, color=c, label=lab)
    ax.set_xticks(x, [short[m] for m in models], fontsize=7.5)
    ax.set_ylabel("Macro-F1, %")
    ax.set_ylim(0, 100)
    grid(ax, "y")
    ax.legend(fontsize=7.5, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.savefig("figures/fig_domain_shift.png")
    plt.close(fig)


def fig_confusion():
    import glob
    panels = [("bert", "BERT (fine-tuned)"), ("qwen2.5-1.5b_zeroshot", "Qwen2.5-1.5B, zero-shot"),
              ("qwen2.5-1.5b_fewshot", "Qwen2.5-1.5B, 6-shot")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, (m, title) in zip(axes, panels):
        cm = 0
        for f in glob.glob(f"results/preds/{m}__pb_test__seed*.csv"):   # summed over all runs
            d = pd.read_csv(f)
            cm = cm + confusion_matrix(d["gold"], d["pred"], labels=LABELS)
        cm = cm / cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), ["neg", "neu", "pos"])
        ax.set_yticks(range(3), ["neg", "neu", "pos"])
        ax.set_xlabel("predicted")
        ax.set_title(title, fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("gold")
    fig.tight_layout()
    fig.savefig("figures/fig_confusion.png")
    plt.close(fig)


def fig_case_study():
    shares = pd.read_csv("results/case_firm_shares.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    left = np.zeros(len(shares))
    y = np.arange(len(shares))[::-1]
    for lab in LABELS:
        v = shares[lab].values * 100
        ax.barh(y, v, left=left, height=0.6, color=SENT_COL[lab], label=lab, edgecolor="white", linewidth=1.5)
        for yi, l, vv in zip(y, left, v):
            if vv >= 8:
                ax.text(l + vv / 2, yi, f"{vv:.0f}%", ha="center", va="center", fontsize=7.5,
                        color="white" if lab != "neutral" else INK)
        left += v
    ax.set_yticks(y, [f"{f}  (n={int(n)})" for f, n in zip(shares.index, shares["n"])])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of headlines, % (FinBERT, average of 5 runs)")
    ax.legend(ncol=3, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 1.2))
    for s in ["left", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    fig.savefig("figures/fig_case_shares.png")
    plt.close(fig)

    yearly = pd.read_csv("results/case_yearly.csv")
    yearly = yearly[(yearly["year"] >= 2011) & (yearly["year"] <= 2020)]
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    for firm, c in [("JPMorgan", BLUE), ("Goldman Sachs", ORANGE), ("BlackRock", AQUA)]:
        d = yearly[yearly["firm"] == firm]
        ax.plot(d["year"], d["net"], color=c, lw=2, marker="o", ms=4.5, markeredgecolor="white",
                markeredgewidth=1, label=firm)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("Net sentiment\n(share pos. - share neg.)")
    ax.set_xticks(range(2011, 2021))
    grid(ax, "y")
    ax.legend(fontsize=7.5, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.15))
    fig.savefig("figures/fig_case_yearly.png")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    for f in [fig_main_results, fig_agreement, fig_domain_shift, fig_confusion, fig_case_study]:
        try:
            f()
            print("ok:", f.__name__)
        except FileNotFoundError as e:
            print("skipped", f.__name__, "-", e)
