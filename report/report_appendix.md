[[PAGEBREAK]]

# Appendix A. Regular expressions

[[TAB:regex]]

# Appendix B. Prompts

Both prompts are sent as a user message through the Qwen chat template (with its default system message). In the 6-shot setting the six demonstrations come first as earlier user/assistant turns, and the model answers with the label word.

**Sentence-level prompt (zero-shot and 6-shot):** *You are a financial analyst. Read the sentence from a financial news text and decide how it would most likely affect the stock price of the company it talks about, from the point of view of an investor. Answer with exactly one word: positive, negative or neutral. Sentence: {text}*

**Firm-aware prompt (case study):** *You are a financial analyst. Read the news headline and decide how it would most likely affect {firm} (its business or its stock price). Be careful: very often {firm} is only mentioned because its analysts rate some other company, or because it owns shares of another company, or the name is used for something else. In such cases the news is neutral for {firm}. Answer with exactly one word: positive, negative or neutral. Headline: {text}*

# Appendix C. Code

All code is in the *code* folder and the README there gives the order in which the scripts are run. *prepare_data.py* builds the splits, *baselines.py*, *neural_models.py*, *finetune_bert.py* and *llm_prompting.py* train or prompt the models and save one prediction file per model, dataset and seed, *evaluate.py* computes every number in Section 6 from these files, *case_study.py* does Section 7 and *make_figures.py* draws the figures.
