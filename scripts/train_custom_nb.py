# scripts/train_custom_nb.py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

from BayesTextClassification import BayesTextClassification
from BayesTextClassificationVectorized import BayesTextClassificationVectorized
from IMDBLoader import IMDBLoader


def main():
    data_loader = IMDBLoader()
    X_train, y_train, X_test, y_test = data_loader.get_splits()

    alpha = 0.1

    # 1) Raw-text custom NB
    custom_raw = BayesTextClassification(alpha=alpha)
    custom_raw.fit(X_train, y_train)
    y_pred_raw = custom_raw.predict(X_test)

    print("=== Custom BayesTextClassification (raw text) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_raw))
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred_raw, digits=4))

    # 2) Custom NB + CountVectorizer
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        token_pattern=r"(?u)\b\w\w+\b",
        lowercase=True,
    )

    custom_cv = BayesTextClassificationVectorized(alpha=alpha, vectorizer=vectorizer)
    custom_cv.fit(X_train, y_train)
    y_pred_cv = custom_cv.predict(X_test)

    print("\n=== Custom BayesTextClassificationVectorized (CountVectorizer) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_cv))
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred_cv, digits=4))


if __name__ == "__main__":
    main()
