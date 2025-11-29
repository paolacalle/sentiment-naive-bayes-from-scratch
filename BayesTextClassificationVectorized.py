import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class BayesTextClassificationVectorized:
    def __init__(self, alpha=1.0, vectorizer=None):
        if alpha < 0:
            raise ValueError("Alpha must be positive.")
        self.alpha = alpha
        
        self.vectorizer = vectorizer or CountVectorizer(
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8,
            token_pattern=r"(?u)\b\w\w+\b", # at least 2 characters
            lowercase=True
        )
        
        self.class_log_prior_ = None      # shape (n_classes,)
        self.feature_log_prob_ = None     # shape (n_classes, n_features)

    def fit(self, X_text, y):
        # 1) Vectorize text -> counts
        X_counts = self.vectorizer.fit_transform(X_text)  # (n_samples, n_features)
        y = np.array(y)
        n_samples, n_features = X_counts.shape

        # 2) Classes and priors
        self.classes_, y_indices = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)

        class_count = np.bincount(y_indices)                     # (n_classes,)
        self.class_log_prior_ = np.log(class_count / n_samples)  # log P(y)

        # 3) Feature counts per class
        #    class_feature_count[c, j] = sum of counts of feature j in class c
        class_feature_count = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, c in enumerate(self.classes_):
            X_c = X_counts[y == c]
            # sum over samples, keep features dimension
            class_feature_count[idx, :] = X_c.sum(axis=0)

        # 4) Apply Laplace smoothing and compute log P(word | class)
        smoothed_fc = class_feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)  # total words per class + alpha*V

        self.feature_log_prob_ = np.log(smoothed_fc / smoothed_cc)  # (n_classes, n_features)

        return self

    def predict(self, X_text):
        # 1) Vectorize new data using *same* vectorizer
        X_counts = self.vectorizer.transform(X_text)  # (n_samples, n_features)

        # 2) Joint log likelihood:
        #    jll = log P(y) + sum_j x_ij * log P(word_j | y)
        jll = X_counts @ self.feature_log_prob_.T      # (n_samples, n_classes)
        jll = jll + self.class_log_prior_              # broadcast add

        # 3) Pick argmax over classes
        indices = np.argmax(jll, axis=1)
        return self.classes_[indices]
