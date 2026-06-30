# Multi-Metric Evaluation of Translation-Based Cross-Lingual Sentiment Consistency Using Large Language Models and Neural Machine Translation

This repository contains the implementation scripts used in the experiments reported in the manuscript:

**Multi-Metric Evaluation of Translation-Based Cross-Lingual Sentiment Consistency Using Large Language Models and Neural Machine Translation**

The repository includes the complete experimental pipeline for multilingual translation, sentiment evaluation, semantic similarity analysis, translation quality assessment, and statistical analysis.

---

# Overview

Cross-lingual sentiment analysis frequently relies on machine translation to overcome language barriers. However, translation may alter sentiment while preserving semantic meaning. This study systematically evaluates whether modern translation systems preserve sentiment consistency across multiple languages.

The benchmark compares ten contemporary translation systems, including commercial Neural Machine Translation (NMT) services, open-source NMT models, commercial Large Language Models (LLMs), general-purpose open-source LLMs, and a specialized translation-oriented LLM.

The experiments include four languages:

- English (EN)
- Chinese (ZH)
- Spanish (ES)
- French (FR)

All twelve possible translation directions were evaluated.

---

# Translation Systems

## Commercial Neural Machine Translation

- Google Translate (Google Cloud Translation API v2)
- Microsoft Translator (Microsoft Translator Text API)

## Open-Source Neural Machine Translation

- Meta NLLB-200-distilled-600M
- LibreTranslate (v1.5)

## Commercial Large Language Models

- GPT-4o-mini
- Gemini 2.5 Flash-Lite

## Open-Source General-Purpose Large Language Models

- Meta-Llama-3.1-8B-Instruct
- Qwen2.5-7B-Instruct
- Mistral-7B-Instruct-v0.3

## Translation-Oriented Large Language Model

- NiuTrans/LMT-60-1.7B

---

# Evaluation Framework

## Sentiment Analysis

Sentiment labels were generated using the multilingual XLM-RoBERTa sentiment classifier:

- CardiffNLP XLM-RoBERTa (twitter-xlm-roberta-base-sentiment)

Additional validation was performed using:

- NLPTown multilingual sentiment classifier

## Sentiment Consistency Metrics

- Accuracy
- Weighted F1-score
- Matthews Correlation Coefficient (MCC)
- Sentiment Stability Ratio (SSR)

## Translation Quality

Translation quality was evaluated using:

- COMET-QE (wmt22-cometkiwi-da)

## Semantic Similarity

Semantic similarity was evaluated using Language-Agnostic BERT Sentence Embeddings (LaBSE). Semantic preservation was quantified by computing cosine similarity between the source and translated sentence embeddings.

---

# Statistical Analysis

Performance differences among translation systems were analyzed using:

- Friedman test
- Nemenyi post-hoc test

The statistical significance threshold was set to **p < 0.05**.

---

# Repository Structure

```
translation/
    Translation scripts for all evaluated translation systems

evaluation/
    Sentiment analysis
    COMET-QE evaluation
    LaBSE semantic similarity
    Evaluation metrics

analysis/
    Statistical analysis

requirements.txt
README.md
```

---

# Computational Environment

The experiments were conducted using:

- Google Colab Pro+
- NVIDIA A100-SXM4-40GB GPU
- PyTorch
- Hugging Face Transformers

---

# Reproducibility

This repository contains the implementation scripts used to reproduce the experiments reported in the associated manuscript.

The original dataset and generated translation outputs are not included in this repository. Users should prepare their own input dataset following the expected schema described in the manuscript.

Commercial API credentials have been removed for security reasons. Users should provide their own API credentials where required.

---

# Citation

Citation information will be added after publication of the manuscript.

---

# License

This project is distributed under the MIT License.
