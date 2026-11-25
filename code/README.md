# Financial news sentiment - code for the Language Technologies homework

Everything was run on a laptop with an Apple M3 chip (16 GB RAM), the transformer models use the Apple GPU (`mps`).
Run all scripts from the project folder (the one that contains `code/` and `data/`), not from `code/`.

```
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt
```

## Order of the scripts

| step | script | what it does | time on M3 |
|---|---|---|---|
| 0 | `download_data.py` | downloads the three datasets into data/ | 1 min |
| 1 | `prepare_data.py` | PhraseBank split, tweets, firm headlines | < 1 min |
| 2 | `baselines.py` | majority, LM lexicon, TF-IDF + LogReg | 1 min |
| 3 | `train_embeddings.py` | fastText on 800k headlines + GloVe download | 5 min |
| 4 | `neural_models.py` | FFNN / CNN / BiLSTM x GloVe / FinText x 5 seeds | ~25 min |
| 5 | `finetune_bert.py bert` and `finetune_bert.py finbert` | 5 seeds each (we ran one seed at a time with a short pause, the laptop gets hot) | ~40 min each |
| 6 | `llm_prompting.py <model> zero` / `few <seed>` | Qwen2.5 prompting, 0.5B / 1.5B / 3B | 4 min (1.5B), 8 min (3B) per run |
| 7 | `case_study.py sample` then `case_study.py analyse` | firm headlines | 1 min |
| 8 | `llm_prompting.py <model> target` | sentence prompt vs firm-aware prompt on the annotated sample | 1 min |
| 9 | `evaluate.py`, `make_figures.py` | tables (results/) and figures (figures/) | 1 min |

`lora_llm.py` (LoRA fine-tuning of Qwen2.5-1.5B) is not part of the report: one run would take over an hour on the
laptop, so we left it as future work.

`regex_rules.py` holds all regular expressions (firm names, noise filters for headlines, number
normalisation, tweet cleaning) and `utils.py` has small shared helpers.

Every model writes its predictions to `results/preds/<model>__<dataset>__seed<k>.csv`, and
`evaluate.py` computes all metrics from these files, so nothing has to be retrained to change a table.

## Data

* Financial PhraseBank v1.0 (Malo et al., 2014) - `FinancialPhraseBank-v1.0.zip` from the
  `takala/financial_phrasebank` repo on Hugging Face, unzip into `data/phrasebank_raw/`. CC BY-NC-SA 3.0.
* Twitter Financial News Sentiment - `sent_train.csv`, `sent_valid.csv` from
  `zeroshot/twitter-financial-news-sentiment`, saved as `data/twitter_sent_train.csv` / `data/twitter_sent_valid.csv`.
* Benzinga / analyst headlines 2009-2020 - the parquet file from `ashraq/financial-news`
  (mirror of the Kaggle "Massive Stock News Analysis DB"), saved as `data/benzinga_headlines.parquet`.

`data/manual_sample_annotated.csv` contains our own annotation of 171 headlines (see the report, section 7).
