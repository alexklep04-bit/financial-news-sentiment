# Builds all the datasets used in the project and saves them as csv files in data/
#   1) Financial PhraseBank -> train / val / test split (64/16/20, stratified)
#   2) Twitter financial news -> used only as an out-of-domain test set
#   3) Benzinga headlines that mention JPMorgan, Goldman Sachs, BlackRock, Vanguard, Citadel
#
# run from the project folder:  python code/prepare_data.py

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

from regex_rules import FIRM_PATTERNS, is_noise_headline, clean_tweet

SEED = 42
RAW = "data/phrasebank_raw/FinancialPhraseBank-v1.0"


def read_phrasebank_file(fname):
    rows = []
    # the files are latin-1, not utf-8 (found out the hard way)
    with open(os.path.join(RAW, fname), encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # label is after the LAST @ (a few sentences contain @ themselves)
            sent, label = line.rsplit("@", 1)
            rows.append((sent.strip(), label.strip()))
    return pd.DataFrame(rows, columns=["text", "label"])


def build_phrasebank():
    full = read_phrasebank_file("Sentences_50Agree.txt")
    print("50Agree file:", len(full), "sentences")

    # every sentence gets the strongest agreement level it belongs to
    full["agreement"] = "50-65%"
    for fname, level in [("Sentences_66Agree.txt", "66-74%"),
                         ("Sentences_75Agree.txt", "75-99%"),
                         ("Sentences_AllAgree.txt", "100%")]:
        s = set(read_phrasebank_file(fname)["text"])
        full.loc[full["text"].isin(s), "agreement"] = level

    n_before = len(full)
    full = full.drop_duplicates(subset="text").reset_index(drop=True)
    print("removed", n_before - len(full), "duplicate sentences")

    # 80% train+val / 20% test, then 80/20 again inside train+val (like in the lectures)
    trval, test = train_test_split(full, test_size=0.2, stratify=full["label"], random_state=SEED)
    train, val = train_test_split(trval, test_size=0.2, stratify=trval["label"], random_state=SEED)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part = part.reset_index(drop=True)
        part.to_csv(f"data/pb_{name}.csv", index=False)
        print(f"  {name:5s} {len(part):5d}", part["label"].value_counts().to_dict())

    print(full["agreement"].value_counts().to_dict())


def build_twitter():
    tw = pd.concat([pd.read_csv("data/twitter_sent_train.csv").assign(split="train"),
                    pd.read_csv("data/twitter_sent_valid.csv").assign(split="valid")], ignore_index=True)
    # we never train on tweets; the official "valid" split is our out-of-domain test set,
    # the tweets that mention one of the five firms are taken from both splits
    tw["label"] = tw["label"].map({0: "negative", 1: "positive", 2: "neutral"})  # bearish/bullish/neutral
    tw["raw_text"] = tw["text"]
    tw["text"] = tw["text"].apply(clean_tweet)
    tw = tw[tw["text"].str.len() > 5].drop_duplicates(subset="text").reset_index(drop=True)

    # which of the five firms does the tweet mention (if any)
    tw["firm"] = ""
    for firm, pat in FIRM_PATTERNS.items():
        hit = tw["raw_text"].str.contains(pat, regex=True)
        tw.loc[hit & (tw["firm"] == ""), "firm"] = firm
    tw.to_csv("data/twitter_test.csv", index=False)
    print("twitter:", len(tw), tw["label"].value_counts().to_dict())
    print("  tweets mentioning one of the firms:", (tw["firm"] != "").sum(),
          tw.loc[tw["firm"] != "", "firm"].value_counts().to_dict())


def build_firm_headlines():
    df = pd.read_parquet("data/benzinga_headlines.parquet")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["headline", "date"])
    df = df[df["date"].dt.year >= 2010]          # a few rows have 1969 dates (unix 0)
    df = df.drop_duplicates(subset="headline")

    parts = []
    for firm, pat in FIRM_PATTERNS.items():
        sub = df[df["headline"].str.contains(pat, regex=True)].copy()
        sub["firm"] = firm
        sub["noise"] = sub["headline"].apply(lambda h: is_noise_headline(h, firm))
        print(f"  {firm:14s} matched {len(sub):5d}   kept {(sub['noise'] == '').sum():5d}")
        parts.append(sub)

    out = pd.concat(parts, ignore_index=True)
    out = out[["date", "firm", "headline", "publisher", "stock", "noise"]]
    out.to_csv("data/firm_headlines_all.csv", index=False)
    kept = out[out["noise"] == ""].reset_index(drop=True)
    kept.drop(columns="noise").to_csv("data/firm_headlines.csv", index=False)
    print("firm headlines kept:", len(kept))
    print(out[out["noise"] != ""]["noise"].value_counts().to_dict())


if __name__ == "__main__":
    build_phrasebank()
    build_twitter()
    build_firm_headlines()
