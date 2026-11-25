# 6 Results

## 6.1 Comparison of model families (RQ1)

Table 1 gives the results on the PhraseBank test set. All trained models beat both baselines. The lexicon's accuracy (57.9%) is even below the majority baseline, although its macro-F1 is almost twice as high: it predicts *neutral* for 613 of the 968 test sentences, and 31% of these are in fact positive or negative, typically sentences that only report numbers, such as "Turnover rose to EUR21m from EUR17m". TF-IDF learns exactly such phrases (its strongest features include *up from*, *rose to* and, after normalisation, *_money_ up*) and reaches 73.9 macro-F1; the regex normalisation adds one point.

[[TAB:main]]

Averaging word vectors works poorly (FFNN, 60-62). Usually only one or two of the 23 words in a sentence decide the label and the average dilutes them; besides, in GloVe the nearest neighbour of *rose* is *fell* (cosine similarity 0.88). The CNN is the best model without a transformer (75.4 ± 0.7) and beats the BiLSTM (72.3 ± 0.4), because the decisive cues are short local phrases. Fine-tuned transformers are another 6-8 points better (BERT 83.5 ± 0.3, FinBERT 81.7 ± 1.1), with the largest gain on the small negative class.

## 6.2 Domain-specific pre-training (RQ2)

On PhraseBank, domain-specific pre-training gave no improvement. FinText is slightly worse than GloVe with all three networks, within the confidence intervals (CNN 74.7 ± 1.0 vs 75.4 ± 0.7), and FinBERT is even worse than BERT (81.7 ± 1.1 vs 83.5 ± 0.3), similar to what Araci [1] observed. PhraseBank is written in plain business English that general text covers well, and with 3,096 labelled sentences both models adapt to the task anyway. On tweets the picture is completely different (Section 6.4).

## 6.3 Where the models fail

Figure 1 shows that accuracy falls steeply with annotator agreement: BERT is right on 97.0% of the sentences with full agreement, but only on 61.6% of those with 50-65% agreement, where the majority baseline already reaches 54.0%. An ensemble of the five BERT runs makes 153 errors; only 14 are on sentences with full agreement, 115 confuse positive and neutral, and only 5 mix up positive and negative. So the models rarely get the direction wrong. The hard part is deciding whether news like a new contract is important enough to count as positive, and here even the annotators disagree.

[[FIG:figures/fig_agreement.png|4.6|Accuracy on the PhraseBank test set by the share of annotators who agreed on the gold label (mean of all runs, n = number of test sentences).]]

## 6.4 Out-of-domain: market tweets (RQ3)

Applied to tweets (Table 2), most models lose 15-25 points: BERT drops from 83.5 to 61.1 macro-F1 and the CNN from 75.4 to 51.3. 26.7% of the word tokens in the tweets never occur in the PhraseBank training set (6.0% for its test set), and market words such as *downgrade*, *beats* or *buyback* are missing completely.

Here domain-specific pre-training pays off. FinText beats GloVe with all three networks (FFNN +5.6, CNN +1.7, BiLSTM +0.6 points), and FinBERT beats BERT by six points (67.0 vs 61.1) and by twelve on the tweets about the five firms (64.7 vs 52.7). The labelled data are the same, so the gain can only come from pre-training on US market language. The best transfer, however, comes from the 6-shot LLM: Qwen2.5-1.5B with PhraseBank demonstrations reaches 72.2 ± 1.6 on the tweets and 70.5 ± 2.3 on the firm tweets.

[[TAB:tweets]]

## 6.5 Prompted LLMs

Without demonstrations the size of the LLM matters (Table 1): Qwen2.5-0.5B answers *neutral* to almost everything (26.6 macro-F1), the 1.5B model reaches 63.4 and the 3B model 73.9, as much as TF-IDF. The 1.5B model finds 99% of the negative and 96% of the positive sentences, but also calls 53% of the neutral ones positive: for it any rise in sales or new contract is good news, while the annotators often label such sentences neutral.

Six demonstrations fix most of this. Qwen2.5-1.5B reaches 80.2 ± 0.5, less than four points below BERT, after seeing 6 labelled sentences instead of 3,096 and without any training, and the share of neutral sentences it calls positive falls to 10%. Surprisingly, the 3B model does not profit at all (70.3 ± 3.3, below its zero-shot 73.9) and still calls about a third of the neutral sentences positive (31% zero-shot, 35% with examples), as if it kept its own idea of good news; with only one prompt we cannot say how general this is. PhraseBank has been public since 2014 and may be part of Qwen's pre-training data, but the weak zero-shot results suggest that the labels were not memorised.

# 7 Case study: news about five financial firms (RQ4)

The firm patterns matched 7,802 headlines from 2010-2020, and the noise filters removed 4,469 of them (2,313 portfolio lists, 1,176 analyst calls, 575 fund products, 405 conference names). Vanguard is hit hardest, only 126 of its 2,233 matches survive. The remaining 3,333 headlines were classified with FinBERT, averaging its 5 runs. For every firm about two thirds are neutral (Figure 2), and the net sentiment (share positive minus share negative) is slightly positive, +3.7 points for JPMorgan, +2.0 for Goldman Sachs and +10.7 for BlackRock. Some yearly dips match real events, such as Goldman Sachs in 2010 (SEC fraud charges) and JPMorgan in 2013 ($13 billion settlement). But many "negative" headlines are not about the firm at all ("Micron Tumbles on Goldman's Bearish Call").

[[FIG:figures/fig_case_shares.png|4.6|Predicted sentiment of the 3,333 filtered headlines per firm (FinBERT, probabilities averaged over 5 runs).]]

We therefore sampled 40 headlines per firm (all 11 for Citadel) and annotated the 171 headlines with three labels: is it news about the firm, the sentiment of the headline (PhraseBank guideline) and the sentiment for the firm. Only 72 headlines (42%) are really about the firm, from 68% for BlackRock to 30% for Goldman Sachs and 20% for Vanguard. Goldman Sachs mostly appears as an analyst in calls our rules missed ("Kellogg +2% after Goldman Sachs bump"), Vanguard as a fund ("It's Time To Buy Vanguard's Energy ETF") or an ordinary word ("... at the Vanguard of the Life Science Ecosystem's Fight"). The sentence and firm labels differ in 59 headlines (35%), once even in opposite directions: "Vanguard poised to overtake BlackRock in Q1 U.S. ETF flows" is good news, but not for BlackRock.

[[TAB:manual]]

Against the sentence labels the prompted LLMs beat FinBERT (Table 3), but against the firm labels all models stay around 50 macro-F1 or below. A firm-aware prompt (Appendix B) helps the LLMs (39.4 to 47.9 for the 1.5B model, 44.6 to 49.5 for the 3B model), yet even when told that analyst ratings of other companies are neutral for the bank, Qwen2.5-3B labels "Vonage +2.7% as JPMorgan sees 25% upside" as positive for JPMorgan. On the 72 headlines that really are about the firm, the agreement is much better (76% for Qwen2.5-3B, 62% for FinBERT). The main problem is therefore not the sentiment itself, but deciding whether the news concerns the firm at all.

# 8 Discussion and limitations

Three results stand out. First, each step of the course from word counts to n-grams, learned representations and pre-trained transformers brought a clear gain, except the averaged-embedding FFNN: how word vectors are combined matters as much as the vectors. Second, domain-specific pre-training brought nothing on PhraseBank but up to twelve points on tweets, and since a real news feed never looks exactly like the training data, the second result matters more in practice. Third, few-shot prompting of a small open LLM is a serious alternative to fine-tuning when labelled data are scarce, although the 3B result shows that the gain is not guaranteed, and it is much slower (almost three minutes for the 968 test sentences with the 3B model, seconds with BERT).

The study has limitations. All numbers come from one fixed split, so the confidence intervals reflect the random seed but not the choice of test sentences. The tweets were labelled by other people with other label definitions. Our LLMs have at most 3 billion parameters, and LoRA [6] fine-tuning was too slow on our laptop. Finally, the case-study headlines were annotated by one person only.

# 9 Conclusions

Our answers to the four research questions are:

- **RQ1.** Fine-tuned BERT is the best model on Financial PhraseBank (83.5 ± 0.3 macro-F1), followed by FinBERT (81.7) and by Qwen2.5-1.5B with six examples in the prompt (80.2), which needed no training. The best model without a transformer, a CNN on GloVe, reaches 75.4 and the finance lexicon 46.9.
- **RQ2.** Domain-specific pre-training did not help on PhraseBank, but clearly helped on tweets (FinBERT +6 to +12 points over BERT, FinText better than GloVe with all networks).
- **RQ3.** Most trained models lose 15-25 points on tweets; the 6-shot LLM (72.2) and FinBERT (67.0) are the most robust.
- **RQ4.** Only 42% of the headlines that mention the firms are really about them, so sentence-level sentiment is a poor measure of firm-level sentiment, and a firm-aware prompt helps only partly.

A natural next step is a two-stage system that first decides whether a headline concerns the firm, for example trained on entity-level data such as SEntFiN [14], and only then judges the sentiment.

# References

[REF] [1] D. Araci. FinBERT: Financial sentiment analysis with pre-trained language models. arXiv:1908.10063, 2019.
[REF] [2] P. Bojanowski, E. Grave, A. Joulin, T. Mikolov. Enriching word vectors with subword information. TACL 5, 135-146, 2017.
[REF] [3] T. Brown et al. Language models are few-shot learners. NeurIPS 2020.
[REF] [4] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL 2019.
[REF] [5] S. Hochreiter, J. Schmidhuber. Long short-term memory. Neural Computation 9(8), 1735-1780, 1997.
[REF] [6] E. J. Hu et al. LoRA: Low-rank adaptation of large language models. ICLR 2022.
[REF] [7] A. H. Huang, H. Wang, Y. Yang. FinBERT: A large language model for extracting information from financial text. Contemporary Accounting Research 40(2), 806-841, 2023.
[REF] [8] Y. Kim. Convolutional neural networks for sentence classification. EMNLP 2014.
[REF] [9] T. Loughran, B. McDonald. When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. Journal of Finance 66(1), 35-65, 2011.
[REF] [10] P. Malo, A. Sinha, P. Korhonen, J. Wallenius, P. Takala. Good debt or bad debt: Detecting semantic orientations in economic texts. JASIST 65(4), 782-796, 2014.
[REF] [11] T. Mikolov, K. Chen, G. Corrado, J. Dean. Efficient estimation of word representations in vector space. arXiv:1301.3781, 2013.
[REF] [12] J. Pennington, R. Socher, C. Manning. GloVe: Global vectors for word representation. EMNLP 2014.
[REF] [13] Qwen Team. Qwen2.5 technical report. arXiv:2412.15115, 2024.
[REF] [14] A. Sinha, S. Kedas, R. Kumar, P. Malo. SEntFiN 1.0: Entity-aware sentiment analysis for financial news. JASIST 73(9), 1314-1335, 2022.
[REF] [15] P. C. Tetlock. Giving content to investor sentiment: The role of media in the stock market. Journal of Finance 62(3), 1139-1168, 2007.
[REF] [16] A. Vaswani et al. Attention is all you need. NeurIPS 2017.
[REF] [17] S. Wu et al. BloombergGPT: A large language model for finance. arXiv:2303.17564, 2023.
[REF] [18] Y. Yang, M. C. S. Uy, A. Huang. FinBERT: A pretrained language model for financial communications. arXiv:2006.08097, 2020.
[REF] [19] W. Zhang, Y. Deng, B. Liu, S. J. Pan, L. Bing. Sentiment analysis in the era of large language models: A reality check. Findings of NAACL 2024.
