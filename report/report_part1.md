# 1 Introduction

Banks and asset managers read far more text than any team of analysts could, and the tone of the news is a standard input for their trading and risk models. Tetlock [15] showed that pessimistic wording in a Wall Street Journal column predicted lower market returns the next day, and Bloomberg even trained its own large language model, BloombergGPT, for financial text [17].

Financial sentiment is harder than it looks. Words such as *liability* or *tax* are negative in everyday language but neutral in a balance sheet [9]. "Operating profit fell to EUR 3.2 mn from EUR 4.5 mn" contains no opinion words, yet every investor reads it as bad news. And in "KBR cut to Sell at Goldman" the news is bad for KBR, while Goldman Sachs is only the messenger.

In this project we compare the model families covered in the Language Technologies course, from a word list to a large language model (LLM), on the same financial sentiment task, and then use them to look at news about JPMorgan, Goldman Sachs, BlackRock, Vanguard and Citadel. We ask four research questions:

- **RQ1.** How do lexicon, bag-of-words, word-embedding (FFNN, CNN, BiLSTM), fine-tuned transformer and prompted LLM approaches compare on financial sentence sentiment?
- **RQ2.** Does domain-specific pre-training help (finance word embeddings instead of general ones, FinBERT instead of BERT)?
- **RQ3.** How well do the models transfer to a different kind of financial text (market tweets)?
- **RQ4.** Can a sentence-level model measure news sentiment *towards* a particular firm?

# 2 Related work

Loughran and McDonald [9] showed that almost three quarters of the "negative" words that a general dictionary finds in 10-K filings are not negative in a financial context, and published finance-specific word lists. The Financial PhraseBank [10] became the standard benchmark for the task, first for n-gram classifiers and later for word embeddings [11, 12, 2] with convolutional [8] and recurrent [5] networks. Araci [1] further pre-trained BERT [4], an encoder built on the Transformer [16], on financial news and reported 86% accuracy on PhraseBank, while Yang et al. [18] and Huang et al. [7] pre-trained BERT from scratch on 4.9 billion tokens of corporate filings, earnings calls and analyst reports; we use this model and call it FinBERT. GPT-3 [3] made few-shot prompting popular, and Zhang et al. [19] found LLMs competitive zero-shot on simple sentiment tasks but behind small fine-tuned models on harder ones. Finally, SEntFiN [14] annotated over 10,000 headlines with one label per mentioned company and found that many contain companies with conflicting sentiment, which is exactly the problem we meet in Section 7.

# 3 Data

**Financial PhraseBank** [10] contains sentences from English news about Finnish listed companies, each labelled by 5-8 annotators with a finance background. We use the version with all sentences where a majority agreed (4,838 sentences after removing duplicates: 59.4% neutral, 28.2% positive, 12.5% negative) and keep the agreement level of every sentence (100%, 75-99%, 66-74% or 50-65%). Following the lectures, the data were split once, stratified by label, into 80% training+validation and 20% test, and the first part again 80/20, giving 3,096 training, 774 validation and 968 test sentences.

**Twitter Financial News** (Hugging Face *zeroshot/twitter-financial-news-sentiment*) contains 11,931 tweets labelled bearish, bullish or neutral, which we map to negative, positive and neutral after removing URLs and the "$" of cashtags. We never train on tweets: the 2,340 tweets of the official validation split are our out-of-domain test set, together with the 125 tweets from both splits that mention one of the five firms.

**Firm headlines.** About 1.8 million unlabelled headlines from 2009-2020 (Benzinga, Seeking Alpha, Zacks and others; Kaggle "Massive Stock News Analysis DB"), 807 thousand after removing duplicates. All of them were used to train our own word embeddings, and the ones about the five firms for the case study.

# 4 Methods

## 4.1 Regular expressions

Regular expressions (Appendix A) are used in three places. (1) **Normalisation for bag-of-words models**: percentages, money amounts, years, quarters and other numbers are replaced with the tokens _pct_, _money_, _year_, _quarter_ and _num_, so "rose to EUR 13.1 mn" and "rose to EUR 2.7 mn" share the same features. (2) **Finding the firms** by their names, CEOs and tickers; negative look-behind and look-ahead exclude namesakes such as "American Vanguard" or "Vanguard Health". (3) **Filtering headlines that are not about the firm**: portfolio lists ("XYZ Advisors Buys iShares ..., Vanguard ..."), analyst calls where the bank is the broker ("cut to Sell at Goldman"), conference names and fund products.

## 4.2 Models

**Baselines and lexicon.** The majority baseline always predicts *neutral*, the random baseline is the sum of squared class probabilities (as in the lectures), and the **Loughran-McDonald lexicon** [9] predicts the polarity with more LM words, or *neutral* on a tie.

**TF-IDF + logistic regression** on word 1-2-grams and character 2-5-grams with balanced class weights; C was chosen on the validation set. We train it with and without the regex normalisation.

**Word embeddings + neural networks.** We compare 100-dimensional **GloVe** [12] vectors with **FinText**, our own fastText skip-gram model [2] trained on the 807k headlines (7.5 million tokens). On top of them we train an FFNN on the averaged, frozen vectors, a CNN in the style of Kim [8] (100 filters of widths 2-5, max-pooling, dropout 0.5) and a BiLSTM [5] with 128 units per direction, using Adam and early stopping on validation macro-F1.

**Fine-tuned transformers.** **BERT-base** [4] and **FinBERT** [7, 18], which has the same architecture but was pre-trained on financial text with its own vocabulary, are fine-tuned with a linear layer on the [CLS] token (AdamW, learning rate 2e-5, batch 16, 4 epochs; the epoch with the best validation macro-F1 is kept). We did not use the popular ProsusAI/finbert, because it was already fine-tuned on PhraseBank and has very likely seen our test sentences.

**Decoder LLMs.** **Qwen2.5-Instruct** [13] with 0.5B, 1.5B and 3B parameters is prompted with the task description from the PhraseBank guideline (how would the news affect the stock price from an investor's point of view), either *zero-shot* or with two random solved training examples per class (*6-shot*, Appendix B). Instead of generating text, we compare the probabilities of *positive*, *negative* and *neutral* as the first answer token, so the answer is always a valid label. Caching the part of the prompt shared by all sentences made 6-shot prompting about ten times faster.

# 5 Experimental setup

All models are trained on the training split, tuned on the validation split and evaluated once on the test split. Every model that depends on random initialisation is trained 5 times with different seeds (for 6-shot prompting: 5 different sets of demonstrations), and we report the mean with a 95% confidence interval

[[EQ:CI = x̄ ± 1.96 · s / √n,   n = 5]]

where s is the sample standard deviation. The lexicon, TF-IDF and zero-shot prompting are deterministic and were run once. Because the classes are imbalanced, the main metric is **macro-averaged F1**; accuracy must beat the majority (59.4%) and random (44.7%) baselines. Everything ran on a laptop with an Apple M3 chip and 16 GB of memory.
