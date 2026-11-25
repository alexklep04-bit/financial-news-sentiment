# Looks at the mistakes of a fine-tuned model (probabilities averaged over the 5 runs)
# and compares it with the LLM and with the lexicon on a few sentences.
#   python code/error_analysis.py bert   (or finbert)
import sys
import pandas as pd

from utils import LABELS
from case_study import ensemble

test = pd.read_csv("data/pb_test.csv")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "bert"
texts, probs = ensemble(MODEL, "pb_test")
assert (texts == test["text"].values).all()
test["model"] = [LABELS[i] for i in probs.argmax(1)]
test["conf"] = probs.max(1)

lex = pd.read_csv("results/preds/lm_lexicon__pb_test__seed0.csv")
test["lexicon"] = lex["pred"].values
try:
    few = pd.read_csv("results/preds/qwen2.5-1.5b_fewshot__pb_test__seed1.csv")
    test["qwen_6shot"] = few["pred"].values
except FileNotFoundError:
    test["qwen_6shot"] = "-"

err = test[test["model"] != test["label"]]
print(f"{MODEL} ensemble: {len(err)} errors out of {len(test)} ({len(err) / len(test):.1%})")
print("errors by agreement level:", err["agreement"].value_counts().to_dict())
print("error types (gold -> pred):", (err["label"] + " -> " + err["model"]).value_counts().to_dict())

pd.set_option("display.max_colwidth", 160)
pd.set_option("display.width", 250)
print("\nmost confident errors:")
print(err.sort_values("conf", ascending=False)[["text", "label", "model", "conf", "agreement"]].head(12).to_string())

print("\nsentences with no opinion words that the model gets right but the lexicon does not:")
m = (test["model"] == test["label"]) & (test["lexicon"] == "neutral") & (test["label"] != "neutral")
print(test[m][["text", "label"]].head(6).to_string())
print(f"({m.sum()} such sentences)")

if (test["qwen_6shot"] != "-").any():
    m = (test["model"] == test["label"]) & (test["qwen_6shot"] != test["label"])
    print(f"\n{MODEL} right, Qwen 6-shot wrong: {m.sum()}; gold->qwen:",
          (test[m]["label"] + " -> " + test[m]["qwen_6shot"]).value_counts().to_dict())
    print(test[m][["text", "label", "qwen_6shot"]].head(8).to_string())
