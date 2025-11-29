# Sentiment Naive Bayes from Scratch

> IMDB movie review sentiment classification using both scikit-learn's Multinomial Naive Bayes and custom, from-scratch implementations.

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)
![Status](https://img.shields.io/badge/status-experimental-orange.svg)

---

## Overview

This repo explores **Naive Bayes for text classification** on the classic **IMDB movie reviews** dataset.  

It includes:

- A **from-scratch implementation** of Multinomial Naive Bayes over raw text
- A **CountVectorizer-based implementation** that closely matches `sklearn.MultinomialNB`
- Comparisons against scikit-learn’s Naive Bayes models
- Experiments on the effect of **Laplace/Lidstone smoothing (α)**  
- Visualizations of the **most positive/negative words** according to the model

The goal is to understand **how Naive Bayes actually works under the hood**, not just call a library and trust vibes.

---

## Dataset

This project uses the **[IMDB movie reviews dataset](https://huggingface.co/datasets/imdb)** loaded via `datasets`:

- 25,000 training reviews
- 25,000 test reviews
- Binary labels: `0 = negative`, `1 = positive`

The loading logic is wrapped in `IMDBLoader` inside `data_loader.py`.

---

## Key Components

- `BayesTextClassification.py`  
  Custom Naive Bayes implementation using manual tokenization and Laplace smoothing.

- `BayesTextClassificationVetorized.py`  
  Custom Naive Bayes built on top of **scikit-learn's `CountVectorizer`**, designed to match `MultinomialNB`’s behavior.

- `IMDBLoader.py`  
  Contains `IMDBLoader`, which downloads and returns train/test splits:
  ```python
  X_train, y_train, X_test, y_test = IMDBLoader().get_splits()
  ```
