# experiments/alpha_sweep.py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from BayesTextClassification import BayesTextClassification
from BayesTextClassificationVectorized import BayesTextClassificationVectorized
from IMDBLoader import IMDBLoader


def main():
    # Load data
    data_loader = IMDBLoader()
    X_train, y_train, X_test, y_test = data_loader.get_splits()

    # Alphas to test (log-spaced)
    alphas = np.arange(0.01, 10, 0.5)

    sk_accuracies = []
    custom_raw_accuracies = []
    custom_cv_accuracies = []

    # Shared vectorizer for sklearn + CV model
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        token_pattern=r"(?u)\b\w\w+\b",
        lowercase=True,
    )

    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)

    for alpha in alphas:
        print(f"\n=== Alpha = {alpha:.4f} ===")

        # 1) sklearn MultinomialNB
        sk_nb = MultinomialNB(alpha=alpha)
        sk_nb.fit(X_train_counts, y_train)
        y_pred_sk = sk_nb.predict(X_test_counts)
        sk_acc = accuracy_score(y_test, y_pred_sk)
        sk_accuracies.append(sk_acc)
        print(f"sklearn MultinomialNB accuracy: {sk_acc:.4f}")

        # 2) Custom model on raw text
        custom_raw = BayesTextClassification(alpha=alpha)
        custom_raw.fit(X_train, y_train)
        y_pred_raw = custom_raw.predict(X_test)
        raw_acc = accuracy_score(y_test, y_pred_raw)
        custom_raw_accuracies.append(raw_acc)
        print(f"Custom BayesTextClassification accuracy: {raw_acc:.4f}")

        # 3) Custom model using CountVectorizer
        custom_cv = BayesTextClassificationVectorized(alpha=alpha, vectorizer=vectorizer)
        custom_cv.fit(X_train, y_train)
        y_pred_cv = custom_cv.predict(X_test)
        cv_acc = accuracy_score(y_test, y_pred_cv)
        custom_cv_accuracies.append(cv_acc)
        print(f"Custom BayesTextClassificationVectorized accuracy: {cv_acc:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, sk_accuracies, marker="o", label="sklearn MultinomialNB")
    plt.plot(alphas, custom_raw_accuracies, marker="s", label="Custom NB (raw text)")
    plt.plot(alphas, custom_cv_accuracies, marker="^", label="Custom NB + CountVectorizer")

    plt.xlabel("Alpha (Laplace smoothing)")
    plt.ylabel("Accuracy")
    plt.title("Effect of Alpha on Naive Bayes Accuracy (IMDB)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
