import numpy as np

class BayesTextClassification:
    """ 
    A simple implementation of a Naive Bayes classifier for text data.
    Assumes binary classification (labels 0 and 1) and uses Laplace smoothing.
    Parameters:
    ----------
    alpha : float
        Smoothing parameter for Laplace smoothing. Must be positive.
    Methods:
    -------
    fit(X, y):
        Trains the model on the provided data.
    predict(X):
        Predicts labels for the provided data.
        
    Notes:
    -----
    This implementation assumes that the input texts are in English and performs basic text cleaning.
    """
    def __init__(self, alpha=1.0):
        if alpha < 0:
            raise ValueError("Alpha must be positive.")
        self.alpha = alpha

        self.zero_counts = {}   # word -> count in class 0
        self.one_counts = {}    # word -> count in class 1
        self.prior_zero = 0.0
        self.prior_one = 0.0

        self.total_zero_words = 0
        self.total_one_words = 0
        self.vocab = set()
    
    def clean_text(self, text):
        remove = ['.', ',', '!', '?', ';', ':', '"', "'", '(', ')',
                  '[', ']', '{', '}', '-', '_', '/', '\\', '|',
                  '@', '#', '$', '%', '^', '&', '*', '~', '`',
                  '<', '>']
        for char in remove:
            text = text.replace(char, "")
        return text.lower()
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # priors
        zero_mask = (y == 0)
        one_mask = (y == 1)

        n_zero = zero_mask.sum()
        n_one = one_mask.sum()
        n_total = len(y)

        self.prior_zero = n_zero / n_total
        self.prior_one = n_one / n_total

        # counts
        for text, label in zip(X, y):
            words = self.clean_text(text).split()
            if label == 0:
                hist = self.zero_counts
            else:
                hist = self.one_counts
            
            for w in words:
                self.vocab.add(w)
                hist[w] = hist.get(w, 0) + 1

        # total word counts per class
        self.total_zero_words = sum(self.zero_counts.values())
        self.total_one_words = sum(self.one_counts.values())
        self.vocab_size = len(self.vocab)
    
    def _log_prob_word_given_class(self, word, class_label):
        """
        Returns log P(word | Y=class_label) with Laplace smoothing.
        """
        if class_label == 0:
            count = self.zero_counts.get(word, 0)
            denom = self.total_zero_words + self.alpha * self.vocab_size
        else:
            count = self.one_counts.get(word, 0)
            denom = self.total_one_words + self.alpha * self.vocab_size

        num = count + self.alpha
        return np.log(num / denom)
    
    def predict(self, X):
        predictions = []

        for text in X:
            words = self.clean_text(text).split()

            # start with log priors
            log_prob_zero = np.log(self.prior_zero)
            log_prob_one = np.log(self.prior_one)

            for w in words:
                # we add log probabilities because log(a*b) = log(a) + log(b) 
                # we assume word independence given class
                log_prob_zero += self._log_prob_word_given_class(w, 0)
                log_prob_one += self._log_prob_word_given_class(w, 1)

            if log_prob_one > log_prob_zero:
                predictions.append(1)
            else:
                predictions.append(0)
        
        return np.array(predictions)
