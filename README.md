# Financial News Sentiment Analysis: From Word Counts to Large Language Models

Mini research project (experimental) for the Language Technologies course.

The project compares the model families from the course on financial sentence sentiment
(Financial PhraseBank: positive / negative / neutral) and tests them on financial tweets.
A case study then looks at news headlines about JPMorgan, Goldman Sachs, BlackRock, Vanguard and Citadel.

| Model | Macro-F1 on PhraseBank test |
|---|---|
| Loughran-McDonald lexicon | 46.9 |
| TF-IDF + logistic regression (regex normalisation) | 73.9 |
| CNN on GloVe embeddings | 75.4 ± 0.7 |
| BERT-base, fine-tuned | **83.5 ± 0.3** |
| FinBERT, fine-tuned | 81.7 ± 1.1 |
| Qwen2.5-1.5B, 6-shot prompt (no training) | 80.2 ± 0.5 |

## Folders

- `code/`: all scripts, see `code/README.md` for the order to run them
- `data/`: processed splits and the 171 hand-annotated headlines (raw downloads are not included, `code/download_data.py` gets them)
- `results/`: predictions of every model and run (`results/preds/`), summary tables and logs
- `figures/`: figures used in the report and slides
- `report/`: the report (Word and PDF)
- `slides/`: the presentation

To recompute every number of the report from the saved predictions:

```
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
python code/evaluate.py
```
