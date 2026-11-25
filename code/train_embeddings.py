# Two sets of word embeddings for the neural models:
#   general   - GloVe 100d (Wikipedia + Gigaword), downloaded through gensim
#   financial - our own fastText skip-gram model trained on ~800k Benzinga / Seeking Alpha headlines
import re
import time
import numpy as np
import pandas as pd
import gensim.downloader as api
from gensim.models import FastText

from regex_rules import normalize

TOKEN_RE = re.compile(r"_\w+_|[a-z]+(?:'[a-z]+)?|[%$€£]")


def tokenize(text):
    return TOKEN_RE.findall(normalize(text))


if __name__ == "__main__":
    heads = pd.read_parquet("data/benzinga_headlines.parquet")["headline"].dropna().drop_duplicates()
    # the PhraseBank test sentences are NOT used here, only news headlines
    corpus = [tokenize(h) for h in heads]
    corpus = [c for c in corpus if len(c) >= 3]
    print("sentences:", len(corpus), " tokens:", sum(len(c) for c in corpus))

    t0 = time.time()
    ft = FastText(vector_size=100, window=5, min_count=3, sg=1, negative=10, workers=8, epochs=8, seed=1,
                  bucket=200_000)   # default 2M buckets gives an 800MB file, too much for a laptop
    ft.build_vocab(corpus_iterable=corpus)
    ft.train(corpus_iterable=corpus, total_examples=len(corpus), epochs=ft.epochs)
    print(f"fastText trained in {time.time() - t0:.0f}s, vocab = {len(ft.wv)}")
    ft.wv.save("data/fin_fasttext_100.kv")

    glove = api.load("glove-wiki-gigaword-100")
    glove.save("data/glove_100.kv")

    # look at some neighbours, nice for the report
    for w in ["bullish", "downgrade", "profit", "layoffs", "goldman", "dividend"]:
        print(f"{w:10s} fin:  ", [x for x, _ in ft.wv.most_similar(w, topn=6)])
        if w in glove:
            print(f"{'':10s} glove:", [x for x, _ in glove.most_similar(w, topn=6)])
