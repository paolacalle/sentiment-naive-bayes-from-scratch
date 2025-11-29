# experiments/vocab_importance.py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from IMDBLoader import IMDBLoader


def show_top_words(words, scores, title, top_n=20):
    print(f"\n{title}")
    for w, s in zip(words, scores):
        print(f"{w:25s} {s:.4f}")


def main():
    # Load data
    data_loader = IMDBLoader()
    X_train, y_train, X_test, y_test = data_loader.get_splits()

    # Vectorizer
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        token_pattern=r"(?u)\b\w\w+\b",
        lowercase=True,
    )

    X_train_counts = vectorizer.fit_transform(X_train)

    # Train NB
    alpha = 0.1
    nb = MultinomialNB(alpha=alpha)
    nb.fit(X_train_counts, y_train)

    feature_names = np.array(vectorizer.get_feature_names_out())
    class_labels = nb.classes_

    # Assume binary: 0 = neg, 1 = pos
    neg_idx = np.where(class_labels == 0)[0][0]
    pos_idx = np.where(class_labels == 1)[0][0]

    log_prob = nb.feature_log_prob_
    log_prob_neg = log_prob[neg_idx]
    log_prob_pos = log_prob[pos_idx]

    # Log-odds: how much more indicative of positive than negative
    log_odds = log_prob_pos - log_prob_neg

    top_n = 20

    # Most positive-indicative
    top_pos_idx = np.argsort(log_odds)[-top_n:]
    top_pos_words = feature_names[top_pos_idx][::-1]
    top_pos_scores = log_odds[top_pos_idx][::-1]

    # Most negative-indicative
    top_neg_idx = np.argsort(log_odds)[:top_n]
    top_neg_words = feature_names[top_neg_idx]
    top_neg_scores = log_odds[top_neg_idx]

    show_top_words(top_pos_words, top_pos_scores,
                   "Most POSITIVE-indicative words (log-odds)")
    show_top_words(top_neg_words, top_neg_scores,
                   "Most NEGATIVE-indicative words (log-odds)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Positive
    axes[0].barh(range(top_n), top_pos_scores, tick_label=top_pos_words)
    axes[0].set_title("Most positive-indicative tokens")
    axes[0].set_xlabel("log P(word|pos) - log P(word|neg)")
    axes[0].invert_yaxis()

    # Negative
    axes[1].barh(range(top_n), top_neg_scores, tick_label=top_neg_words)
    axes[1].set_title("Most negative-indicative tokens")
    axes[1].set_xlabel("log P(word|pos) - log P(word|neg)")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
