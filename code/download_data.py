# downloads the three datasets from the Hugging Face hub into data/
import os
import shutil
import zipfile
from huggingface_hub import hf_hub_download

os.makedirs("data", exist_ok=True)

# Financial PhraseBank (the HF repo only has a loading script + the original zip)
p = hf_hub_download("takala/financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset")
with zipfile.ZipFile(p) as z:
    for name in z.namelist():
        if name.startswith("FinancialPhraseBank-v1.0/") and not name.endswith("/"):
            z.extract(name, "data/phrasebank_raw")

# tweets
for f in ["sent_train.csv", "sent_valid.csv"]:
    p = hf_hub_download("zeroshot/twitter-financial-news-sentiment", f, repo_type="dataset")
    shutil.copy(p, f"data/twitter_{f}")

# ~1.8M headlines with dates and tickers (2009-2020)
p = hf_hub_download("ashraq/financial-news", "data/train-00000-of-00001-8ec327f23bbe0948.parquet", repo_type="dataset")
shutil.copy(p, "data/benzinga_headlines.parquet")
print("done")
