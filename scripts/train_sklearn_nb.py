# scripts/train_sklearn_nb.py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from IMDBLoader import IMDBLoader


def main():
    data_loader = IMDBLoader()
    X_train, y_train, X_test, y_test = data_loader.get_splits()

    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        token_pattern=r"(?u)\b\w\w+\b",
        lowercase=True,
    )

    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)

    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_counts, y_train)

    y_pred = nb.predict(X_test_counts)

    print("=== sklearn MultinomialNB (CountVectorizer) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    main()
