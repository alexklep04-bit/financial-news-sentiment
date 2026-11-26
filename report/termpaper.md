# 1 INTRODUCTION

Banks and asset managers read far more text than any team of analysts could, and the tone of the news is a standard input for their trading and risk models. Tetlock [@tetlock] showed that pessimistic wording in a Wall Street Journal column predicted lower market returns the next day, and Bloomberg even trained its own large language model for financial text, BloombergGPT [@wu].

## 1.1 Problem definition

Financial sentiment analysis is harder than general sentiment analysis for three reasons. First, many words have a special meaning in finance: *liability* or *tax* are negative in everyday language but neutral in a balance sheet [@lm]. Second, a lot of bad news contains no opinion words at all: "Operating profit fell to EUR 3.2 mn from EUR 4.5 mn" is clearly negative for an investor, but a word list finds nothing in it. Third, a headline that mentions a company is often not about that company: in "KBR cut to Sell at Goldman" the news is bad for KBR, while Goldman Sachs is only the bank whose analyst made the call.

The Language Technologies course covered several families of methods, from word lists to large language models (LLM). It is not obvious which of them handles these problems best, how much finance-specific pre-training helps, and whether the results can be used to measure the news sentiment about a particular firm.

## 1.2 Goal of the paper

The goal of this work is to compare the text classification methods covered in the course on financial news sentiment under one evaluation protocol, and to find out whether they can measure the news sentiment towards specific financial firms. The work answers four research questions (RQ):

1. **RQ1.** How do lexicon, bag-of-words, word-embedding (FFNN, CNN, BiLSTM), fine-tuned transformer and prompted LLM approaches compare on financial sentence sentiment?
2. **RQ2.** Does domain-specific pre-training help, i.e. finance word embeddings instead of general ones and FinBERT instead of BERT?
3. **RQ3.** How well do the models transfer to a different kind of financial text, market tweets?
4. **RQ4.** Can a sentence-level model measure the news sentiment towards a particular firm, here JPMorgan, Goldman Sachs, BlackRock, Vanguard and Citadel?

## 1.3 Tasks

To reach the goal, the following tasks were set:

1. Review the literature on financial sentiment analysis and the methods covered in the course.
2. Prepare the datasets: split the Financial PhraseBank, clean the tweets, and use regular expressions to normalise numbers and to find and filter the headlines about the five firms.
3. Implement and train the models of five families with the same data split and the same evaluation protocol.
4. Evaluate the models on the test set, by annotator agreement, and on tweets that were not used for training.
5. Apply the models to news about the five firms and check the results on a manually annotated sample of headlines.

# 2 ANALYSIS OF LITERATURE AND METHODS

## 2.1 Literature analysis

**Lexicons.** Loughran and McDonald [@lm] showed that almost three quarters of the "negative" words that a general dictionary finds in 10-K filings are not negative in a financial context, and published finance-specific word lists that are still widely used.

**Supervised models.** The Financial PhraseBank of Malo et al. [@malo] became the standard benchmark for financial sentence sentiment. Early approaches used n-gram features with linear classifiers, later work used word embeddings [@mikolov, @pennington, @bojanowski] with convolutional [@kim] and recurrent [@hochreiter] networks.

**Transformers.** BERT [@devlin] is an encoder built on the Transformer architecture [@vaswani] and pre-trained on general English text. Araci [@araci] further pre-trained it on financial news and reported 86% accuracy on PhraseBank. Yang et al. [@yang] and Huang et al. [@huang] pre-trained BERT from scratch on 4.9 billion tokens of corporate filings, earnings calls and analyst reports with a finance-specific vocabulary; this model is called FinBERT in this paper.

**Large language models.** GPT-3 [@brown] made few-shot prompting popular: the task and a few solved examples are written in the prompt, and no weights are updated. Zhang et al. [@zhang] found LLMs competitive zero-shot on simple sentiment tasks but behind small fine-tuned models on harder ones. Fine-tuning a whole LLM is expensive, so parameter-efficient methods such as Low-Rank Adaptation (LoRA) [@hu] train only small low-rank matrices. Qwen2.5 [@qwen] is a recent family of open LLMs that is available in sizes small enough for a laptop.

**Entity-level sentiment.** SEntFiN [@sinha] annotated over 10,000 headlines with one label per mentioned company and found that many of them contain companies with conflicting sentiment. This is the problem studied in the case study of Chapter 5.

## 2.2 Methods analysis

This section describes the five model families compared in this work, from the simplest to the most complex; their exact settings are given in Section 3.3.

**Baselines and lexicon.** The majority baseline always predicts the most frequent class, *neutral*. The random baseline is the expected accuracy of guessing with the class proportions p, as defined in the evaluation lecture:

[[EQ:random|Acc_{random} = Σ_{c} p_{c}^{2}]]

The Loughran-McDonald (LM) lexicon [@lm] counts the positive and negative finance words in a sentence and predicts the polarity with more words, or *neutral* on a tie.

**TF-IDF and logistic regression.** Each sentence is turned into a vector of term frequency - inverse document frequency (TF-IDF) weights of word 1-2-grams and character 2-5-grams. The weight of a term t in a sentence d is computed with {eq:tfidf}:

[[EQ:tfidf|w(t, d) = (1 + ln tf(t, d)) · (ln((1 + N) / (1 + df(t))) + 1)]]

where tf(t, d) is the number of times t occurs in d, N is the number of training sentences and df(t) the number of sentences that contain t. Common words such as *the* get a low weight, while terms that are typical of a sentence but rare in the collection get a high one. A logistic regression (LR) with balanced class weights then learns one weight per term and class.

**Word embeddings and neural networks.** Word embeddings represent each word as a dense vector, so that words used in similar contexts get similar vectors. We compare general GloVe vectors [@pennington], trained on Wikipedia and Gigaword, with FinText, our own fastText skip-gram model [@bojanowski] trained on financial headlines. fastText also uses character n-grams, so it can build vectors even for words it has never seen. Three networks use these vectors: a feed-forward neural network (FFNN) on the averaged vectors of a sentence; a convolutional neural network (CNN) in the style of Kim [@kim], whose filters of width 2-5 work as learned n-gram detectors followed by max-pooling; and a bidirectional long short-term memory network (BiLSTM) [@hochreiter], which reads the sentence word by word in both directions and keeps a memory controlled by gates.

**Fine-tuned transformers.** BERT [@devlin] uses self-attention, so the representation of each word depends on all other words of the sentence. For classification, a linear layer is put on top of the special classification token (CLS) and the whole model is fine-tuned on the labelled data. FinBERT [@yang, @huang] has the same architecture but was pre-trained on financial text with its own vocabulary. We deliberately did not use the popular ProsusAI/finbert checkpoint, because it was already fine-tuned on PhraseBank and has very likely seen our test sentences.

**Prompted LLMs.** Qwen2.5-Instruct [@qwen] is a decoder-only model that predicts the next token; it is not trained at all in this work. The *zero-shot* prompt contains the task description from the PhraseBank annotation guideline (how would the news affect the stock price from an investor's point of view) and the sentence, and the *6-shot* prompt adds two solved training examples of each class (Appendix B). Instead of generating free text, we compare the probabilities of *positive*, *negative* and *neutral* as the next token, so the answer is always a valid label.

# 3 DATA AND EXPERIMENTAL SETUP

## 3.1 Datasets

**Financial PhraseBank** [@malo] contains sentences from English news about Finnish listed companies, each labelled by 5-8 annotators with a finance background. We use the version with all sentences on which a majority of annotators agreed (4,838 sentences after removing duplicates: 59.4% neutral, 28.2% positive, 12.5% negative) and keep the agreement level of every sentence (100%, 75-99%, 66-74% or 50-65%). Following the evaluation lecture, the data were split once, stratified by label, into 80% training+validation and 20% test, and the first part again 80/20 ({tab:data}). All models use exactly the same split.

**Twitter Financial News** (Hugging Face *zeroshot/twitter-financial-news-sentiment*) contains 11,931 tweets labelled bearish, bullish or neutral, which we map to negative, positive and neutral after removing URLs and the "$" of cashtags. No model is trained on tweets: the 2,340 tweets of the official validation split are an out-of-domain test set, together with the 125 tweets from both splits that mention one of the five firms.

**Financial headlines.** About 1.8 million unlabelled headlines from 2009-2020 (Benzinga, Seeking Alpha, Zacks and other sites; Kaggle "Massive Stock News Analysis DB"), 807 thousand after removing duplicates. All of them were used to train the FinText embeddings, and the ones about the five firms for the case study.

[[TAB:data]]

## 3.2 Regular expressions

Regular expressions (Appendix A, {tab:regex}) are used in three places. (1) **Normalisation for the bag-of-words and embedding models.** More than half of the training sentences contain numbers, and 1,194 of the 1,637 different number tokens occur only once, so a classifier cannot learn anything from them. Percentages, money amounts, years, quarters and other numbers are therefore replaced with the tokens _pct_, _money_, _year_, _quarter_ and _num_, so that "rose to EUR 13.1 mn" and "rose to EUR 2.7 mn" share the same features. (2) **Finding the firms** by their names, CEOs and tickers; a negative look-behind and look-ahead exclude namesakes such as "American Vanguard" or "Vanguard Health". (3) **Filtering headlines that are not about the firm:** portfolio lists ("XYZ Advisors Buys iShares ..., Vanguard ..."), analyst calls where the bank is only the broker ("cut to Sell at Goldman"), conference names and fund products.

## 3.3 Model configuration

{tab:hyper} lists the main settings of all models. The FinText embeddings have 100 dimensions, like the GloVe vectors, and were trained on 7.5 million tokens with a window of 5, 10 negative samples and 8 epochs. The only hyperparameter tuned by a search is C of the logistic regression; for the neural networks and BERT we used standard values from the literature and selected the training length on the validation set. Caching the part of the LLM prompt that is the same for all sentences (key-value (KV) cache) made 6-shot prompting about ten times faster with identical predictions. All experiments ran on a laptop with an Apple M3 chip, 16 GB of memory and its integrated graphics processing unit (GPU).

[[TAB:hyper]]

## 3.4 Evaluation methodology

All models are trained on the training split, tuned on the validation split and evaluated once on the test split. For every class c, precision, recall and F1 are computed from the true positives (TP), false positives (FP) and false negatives (FN) with {eq:pr} and {eq:f1}:

[[EQ:pr|P_{c} = TP_{c} / (TP_{c} + FP_{c}),     R_{c} = TP_{c} / (TP_{c} + FN_{c})]]

[[EQ:f1|F1_{c} = 2 · P_{c} · R_{c} / (P_{c} + R_{c})]]

Because the classes are imbalanced, the main metric is macro-averaged F1 ({eq:macro}), which gives every class the same weight:

[[EQ:macro|Macro-F1 = (1 / |C|) · Σ_{c ∈ C} F1_{c}]]

Accuracy is reported as well; it is only meaningful if it beats the majority (59.4%) and random (44.7%, see {eq:random}) baselines. Every model that depends on random initialisation is trained 5 times with different seeds (for 6-shot prompting, with 5 different sets of demonstrations), and we report the mean with a 95% confidence interval (CI) computed with {eq:ci}:

[[EQ:ci|CI = x̄ ± 1.96 · s / √n,     n = 5]]

where x̄ and s are the mean and the sample standard deviation of the 5 scores. If the intervals of two models overlap, we do not claim that one of them is better. The lexicon, TF-IDF and zero-shot prompting are deterministic and were run once.

# 4 RESULTS

## 4.1 Comparison of model families (RQ1)

{tab:main} gives the results on the PhraseBank test set. All trained models beat both baselines. The lexicon's accuracy (57.9%) is even below the majority baseline, although its macro-F1 is almost twice as high: it predicts *neutral* for 613 of the 968 test sentences, and 31% of these are in fact positive or negative, typically sentences that only report numbers, such as "Turnover rose to EUR21m from EUR17m". TF-IDF learns exactly such phrases (its strongest features include *up from*, *rose to* and, after normalisation, *_money_ up*) and reaches 73.9 macro-F1; the regex normalisation adds one point.

Averaging word vectors works poorly (FFNN, 60-62). Usually only one or two of the 23 words in a sentence decide the label, and the average dilutes them; besides, in GloVe the nearest neighbour of *rose* is *fell* (cosine similarity 0.88). The CNN is the best model without a transformer (75.4 ± 0.7) and beats the BiLSTM (72.3 ± 0.4), because the decisive cues are short local phrases. Fine-tuned transformers are another 6-8 points better (BERT 83.5 ± 0.3, FinBERT 81.7 ± 1.1), with the largest gain on the small negative class (F1 84.8 for BERT against 73.8 for the CNN).

[[TAB:main]]

## 4.2 Domain-specific pre-training (RQ2)

On PhraseBank, domain-specific pre-training gave no improvement. FinText is slightly worse than GloVe with all three networks, within the confidence intervals (CNN 74.7 ± 1.0 vs 75.4 ± 0.7), and FinBERT is even worse than BERT (81.7 ± 1.1 vs 83.5 ± 0.3, the intervals do not overlap), similar to what Araci [@araci] observed. PhraseBank is written in plain business English that general text covers well, and with 3,096 labelled sentences both models adapt to the task anyway. On tweets the picture is completely different (Section 4.4).

## 4.3 Error analysis

{fig:agreement} shows that accuracy falls steeply with annotator agreement: BERT is right on 97.0% of the sentences with full agreement, but only on 61.6% of those with 50-65% agreement, where the majority baseline already reaches 54.0%. An ensemble of the five BERT runs makes 153 errors. Only 14 of them are on sentences with full agreement, 115 confuse positive and neutral, and only 5 mix up positive and negative. So the models rarely get the direction wrong. The hard part is deciding whether news like a new contract is important enough to count as positive, and here even the annotators disagree.

[[FIG:agreement|figures/fig_agreement.png|4.8|Accuracy on the PhraseBank test set by the share of annotators who agreed on the gold label (mean of all runs, n = number of test sentences)]]

## 4.4 Out-of-domain evaluation on tweets (RQ3)

Applied to tweets ({tab:tweets}), most models lose 15-25 points: BERT drops from 83.5 to 61.1 macro-F1 and the CNN from 75.4 to 51.3. 26.7% of the word tokens in the tweets never occur in the PhraseBank training set (6.0% for its test set), and market words such as *downgrade*, *beats* or *buyback* are missing completely.

[[TAB:tweets]]

Here domain-specific pre-training pays off. FinText beats GloVe with all three networks (FFNN +5.6, CNN +1.7, BiLSTM +0.6 points), and FinBERT beats BERT by six points (67.0 vs 61.1) and by twelve on the tweets about the five firms (64.7 vs 52.7). The labelled data are the same, so the gain can only come from pre-training on US market language. The best transfer, however, comes from the 6-shot LLM: Qwen2.5-1.5B with PhraseBank demonstrations reaches 72.2 ± 1.6 on the tweets and 70.5 ± 2.3 on the tweets about the firms.

## 4.5 Prompted large language models

Without demonstrations the size of the LLM matters ({tab:main}): Qwen2.5-0.5B answers *neutral* to almost everything (26.6 macro-F1), the 1.5B model reaches 63.4 and the 3B model 73.9, as much as TF-IDF. {fig:confusion} shows why the zero-shot 1.5B model is weak: it finds 99% of the negative and 96% of the positive sentences, but it also calls 53% of the neutral ones positive. For the model any rise in sales or new contract is good news, while the annotators often label such sentences neutral.

[[FIG:confusion|figures/fig_confusion.png|5.6|Confusion matrices on the PhraseBank test set (rows: gold label, columns: prediction; normalised per row, summed over all runs)]]

Six demonstrations fix most of this. Qwen2.5-1.5B reaches 80.2 ± 0.5, less than four points below BERT, after seeing 6 labelled sentences instead of 3,096 and without any training, and the share of neutral sentences it calls positive falls to 10%. Surprisingly, the 3B model does not profit at all (70.3 ± 3.3, below its zero-shot 73.9) and still calls about a third of the neutral sentences positive (31% zero-shot, 35% with examples), as if it kept its own idea of good news; with only one prompt we cannot say how general this is. PhraseBank has been public since 2014 and may be part of Qwen's pre-training data, but the weak zero-shot results suggest that the labels were not memorised.

# 5 CASE STUDY: NEWS ABOUT FIVE FINANCIAL FIRMS (RQ4)

## 5.1 Selection of the headlines

The firm patterns matched 7,802 headlines from 2010-2020, and the noise filters removed 4,469 of them (2,313 portfolio lists, 1,176 analyst calls, 575 fund products and 405 conference names). Vanguard is hit hardest: only 126 of its 2,233 matches survive, because its name appears mostly in lists of exchange-traded fund (ETF) holdings. The remaining 3,333 headlines (1,320 JPMorgan, 1,108 Goldman Sachs, 768 BlackRock, 126 Vanguard and 11 Citadel) were classified with FinBERT, averaging the probabilities of its 5 runs.

## 5.2 Sentiment per firm

For every firm about two thirds of the headlines are neutral ({fig:case}), and the net sentiment (share of positive minus share of negative headlines) is slightly positive: +3.7 points for JPMorgan, +2.0 for Goldman Sachs and +10.7 for BlackRock. Some yearly dips match real events, such as Goldman Sachs in 2010 (the fraud charges of the U.S. Securities and Exchange Commission, SEC) and JPMorgan in 2013 (the planned $13 billion settlement). But many "negative" headlines behind these numbers are not about the firm at all ("Micron Tumbles on Goldman's Bearish Call").

[[FIG:case|figures/fig_case_shares.png|4.8|Predicted sentiment of the 3,333 filtered headlines per firm (FinBERT, probabilities averaged over 5 runs)]]

## 5.3 Manual evaluation

We therefore sampled 40 headlines per firm (all 11 for Citadel) and annotated the 171 headlines with three labels: whether the headline is news about the firm, the sentiment of the headline (PhraseBank guideline) and the sentiment for the firm. Only 72 headlines (42%) are really about the firm, from 68% for BlackRock to 30% for Goldman Sachs and 20% for Vanguard. Goldman Sachs mostly appears as an analyst in calls that our rules missed ("Kellogg +2% after Goldman Sachs bump"), and Vanguard as a fund ("It's Time To Buy Vanguard's Energy ETF") or as an ordinary word ("... at the Vanguard of the Life Science Ecosystem's Fight"). The sentence and firm labels differ in 59 headlines (35%), once even in opposite directions: "Vanguard poised to overtake BlackRock in Q1 U.S. ETF flows" is good news, but not for BlackRock.

[[TAB:manual]]

Against the sentence labels the prompted LLMs beat FinBERT ({tab:manual}), but against the firm labels all models stay around 50 macro-F1 or below. A firm-aware prompt (Appendix B) helps the LLMs (from 39.4 to 47.9 for the 1.5B model and from 44.6 to 49.5 for the 3B model), yet even when told that analyst ratings of other companies are neutral for the bank, Qwen2.5-3B labels "Vonage +2.7% as JPMorgan sees 25% upside" as positive for JPMorgan. On the 72 headlines that really are about the firm the agreement is much better (76% for Qwen2.5-3B, 62% for FinBERT). The main problem is therefore not the sentiment itself, but deciding whether the news concerns the firm at all.

# 6 RESULTS AND CONCLUSIONS

## 6.1 Results

1. Fine-tuned BERT is the best model on the Financial PhraseBank test set with 83.5 ± 0.3 macro-F1, followed by FinBERT (81.7 ± 1.1) and by Qwen2.5-1.5B with six examples in the prompt (80.2 ± 0.5). The best model without a transformer, a CNN on GloVe vectors, reaches 75.4 ± 0.7, TF-IDF 73.9 and the finance lexicon 46.9.
2. Regular expressions improved TF-IDF from 72.9 to 73.9 macro-F1 by normalising numbers, and removed 4,469 of the 7,802 headlines that mention the five firms but are mostly not news about them.
3. Domain-specific pre-training gave no gain on PhraseBank, but on tweets FinBERT was 6 points better than BERT on the validation split and 12 points better on the tweets about the five firms, and FinText beat GloVe with all three networks.
4. Most trained models lose 15-25 points of macro-F1 on tweets; the 6-shot LLM (72.2) and FinBERT (67.0) are the most robust.
5. Six demonstrations raised Qwen2.5-1.5B from 63.4 to 80.2 macro-F1, but did not help Qwen2.5-3B (73.9 zero-shot, 70.3 with six examples).
6. Only 42% of the 171 manually checked firm headlines are really about the firm, and all models reach about 50 macro-F1 or less against the firm-level labels.

## 6.2 Conclusions

1. The ranking of the model families mostly follows the order of the course: each step from word counts to n-grams, learned representations and pre-trained transformers brought a clear gain. The exception is the averaged-embedding FFNN, which shows that how word vectors are combined matters as much as the vectors themselves (RQ1).
2. Few-shot prompting of a small open LLM is a serious alternative to fine-tuning when labelled data are scarce, but the gain from demonstrations is not guaranteed, and prompting is much slower: Qwen2.5-3B needed almost three minutes for the 968 test sentences, the fine-tuned BERT a few seconds (RQ1).
3. Whether domain-specific pre-training helps depends on the test data. It does not help when labelled data of the same kind are available, but it helps a lot when the type of text changes, which is the realistic case for a news feed (RQ2, RQ3).
4. Sentence-level sentiment is a poor measure of the sentiment towards a firm, because most headlines that mention a firm are not about it. Measuring firm-level sentiment needs a step that first decides whether a headline concerns the firm (RQ4).

## 6.3 Limitations and future work

All numbers come from one fixed split, so the confidence intervals reflect the random seed but not the choice of the test sentences. The tweets were labelled by other people with other label definitions, so part of the drop in Section 4.4 comes from the labels and not only from the text. The LLMs have at most 3 billion parameters and were tested with a single prompt, and LoRA [@hu] fine-tuning was too slow on the laptop. The case-study headlines were annotated by one person only. A natural next step is a two-stage system that first decides whether a headline concerns the firm, for example trained on entity-level data such as SEntFiN [@sinha], and only then judges the sentiment.
