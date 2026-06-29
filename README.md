# multilingual-translation-benchmark
Implementation scripts for multilingual translation benchmark experiments.
# Multi-Metric Evaluation of Translation-Based Cross-Lingual Sentiment Consistency Using Large Language Models and Neural Machine Translation

This repository contains the implementation used in the experiments presented in the paper:

**Multi-Metric Evaluation of Translation-Based Cross-Lingual Sentiment Consistency Using Large Language Models and Neural Machine Translation**

The repository provides the scripts used for multilingual translation, sentiment evaluation, semantic similarity analysis, translation quality assessment, and statistical analysis.

---

# Overview

Cross-lingual sentiment analysis often relies on machine translation to bridge language differences. However, translation errors may alter the original sentiment while preserving semantic meaning. This repository accompanies our study, which systematically evaluates whether modern translation systems preserve sentiment consistency across multiple language pairs.

The benchmark compares ten contemporary translation systems, including commercial Neural Machine Translation (NMT) services, open-source NMT models, commercial Large Language Models (LLMs), general-purpose open-source LLMs, and a specialized translation-oriented LLM.

The experiments evaluate four languages:

- English (EN)
- Chinese (ZH)
- Spanish (ES)
- French (FR)

covering all 12 possible translation directions.

---

# Translation Systems

The following translation systems were evaluated:

## Commercial Neural Machine Translation

- Google Translate (Google Cloud Translation API)
- Microsoft Translator (Microsoft Translator Text API)

## Open-Source Neural Machine Translation

- Meta NLLB-200-distilled-600M
- LibreTranslate

## Commercial Large Language Models

- GPT-4o-mini
- Gemini 2.5 Flash-Lite

## Open-Source General-Purpose LLMs

- Meta-Llama-3.1-8B-Instruct
- Qwen2.5-7B-Instruct
- Mistral-7B-Instruct-v0.3

## Translation-Oriented Large Language Model

- NiuTrans/LMT-60-1.7B

---

# Evaluation Framework

The translated outputs were evaluated using a multidimensional framework consisting of sentiment preservation, semantic similarity, and translation quality metrics.

## Sentiment Analysis

- XLM-RoBERTa (cardiffnlp/twitter-xlm-roberta-base-sentiment)
- NLPTown multilingual sentiment model (cross-model validation)

## Sentiment Consistency Metrics

- Accuracy
- Weighted F1-score
- Matthews Correlation Coefficient (MCC)
- Sentiment Stability Ratio (SSR)

## Translation Quality

- COMET-QE (wmt22-cometkiwi-da)

## Semantic Similarity

- LaBSE
- Cosine Similarity

---

# Statistical Analysis

Performance differences among translation systems were evaluated using:

- Friedman Test
- Nemenyi Post-hoc Test

Significance level:

```
p < 0.05
```

---

# Repository Structure

```
translation/
    Commercial and open-source translation scripts

evaluation/
    Sentiment analysis
    COMET-QE
    LaBSE
    Evaluation metrics

analysis/
    Statistical analysis
    Tables
    Figures

requirements.txt

README.md
```

---

# Computational Environment

Experiments were conducted using

- Google Colab Pro+
- NVIDIA A100-SXM4-40GB GPU
- Hugging Face Transformers
- PyTorch

---

# Reproducibility

The repository contains the implementation used to reproduce the experiments reported in the paper.

The provided scripts include:

- multilingual translation
- sentiment prediction
- COMET-QE evaluation
- LaBSE similarity computation
- statistical analysis

Commercial API credentials have been removed for security reasons and must be supplied by users through their own API keys.

---

# Installation

Install the required packages using

```bash
pip install -r requirements.txt
```

---

# Citation

If you use this repository in your research, please cite the associated publication.

Citation information will be updated after publication.

---

# License

This project is released under the MIT License.
