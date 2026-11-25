# Non-neural models:
#   - majority / random baselines (only computed, nothing is trained)
#   - Loughran-McDonald finance lexicon (counting positive vs negative words)
#   - TF-IDF + logistic regression, with and without the regex normalisation
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline, make_union
import pysentiment2 as ps

from utils import LABELS, load_split, save_preds
from regex_rules import normalize

train, val, test = load_split("pb_train"), load_split("pb_val"), load_split("pb_test")
tw = load_split("twitter_test")

# ---- majority and random baselines (formulas from the evaluation lecture) ----
p = train["label"].value_counts(normalize=True)
print("class distribution in train:", p.round(3).to_dict())
print(f"majority baseline = {p.max():.4f}, random baseline = {(p ** 2).sum():.4f}")
p_tw = tw["label"].value_counts(normalize=True)
print(f"twitter: majority = {p_tw.max():.4f}, random = {(p_tw ** 2).sum():.4f}")
for name, d in [("pb_test", test), ("twitter", tw)]:
    save_preds("majority", name, 0, d, ["neutral"] * len(d))

# ---- Loughran & McDonald dictionary ----
lm = ps.LM()


def lm_predict(texts):
    preds = []
    for t in texts:
        s = lm.get_score(lm.tokenize(t))
        if s["Positive"] > s["Negative"]:
            preds.append("positive")
        elif s["Negative"] > s["Positive"]:
            preds.append("negative")
        else:
            preds.append("neutral")
    return preds


for name, d in [("pb_test", test), ("twitter", tw)]:
    save_preds("lm_lexicon", name, 0, d, lm_predict(d["text"]))


# ---- TF-IDF + logistic regression ----
def make_model(C, use_regex):
    prep = normalize if use_regex else str.lower
    words = TfidfVectorizer(preprocessor=prep, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    chars = TfidfVectorizer(preprocessor=prep, analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    return make_pipeline(make_union(words, chars), clf)


for use_regex in [False, True]:
    name = "tfidf_lr" + ("_regex" if use_regex else "")
    # pick C on the validation set
    best_c, best_f1 = None, -1
    for C in [0.3, 1, 3, 10, 30]:
        m = make_model(C, use_regex).fit(train["text"], train["label"])
        f1 = f1_score(val["label"], m.predict(val["text"]), average="macro")
        print(f"   {name} C={C:<5} val macroF1={f1:.4f}")
        if f1 > best_f1:
            best_c, best_f1 = C, f1
    print(f"   -> best C = {best_c}")
    m = make_model(best_c, use_regex).fit(train["text"], train["label"])
    for dname, d in [("pb_test", test), ("twitter", tw)]:
        probs = m.predict_proba(d["text"])
        # sklearn sorts classes alphabetically, which is the same order as LABELS
        assert list(m.classes_) == LABELS
        save_preds(name, dname, 0, d, m.predict(d["text"]), probs)

    if use_regex:
        # top features per class, used in the report
        vec = m.steps[0][1]
        feats = np.array(vec.transformer_list[0][1].get_feature_names_out().tolist() +
                         vec.transformer_list[1][1].get_feature_names_out().tolist())
        coef = m.steps[1][1].coef_
        n_words = len(vec.transformer_list[0][1].get_feature_names_out())
        for i, lab in enumerate(LABELS):
            word_coef = coef[i, :n_words]
            top = np.argsort(word_coef)[::-1][:12]
            print(f"   top words for {lab}: {', '.join(feats[top])}")
