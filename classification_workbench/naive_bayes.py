"""Multinomial naive Bayes over sparse, nonnegative document vectors."""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class SparseMultinomialNB(ClassifierMixin, BaseEstimator):
    def __init__(self, alpha=1.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive.")
        self.alpha = alpha

    def fit(self, features, labels):
        features, labels = check_X_y(
            csr_matrix(features, dtype=np.float64), labels, accept_sparse="csr"
        )
        if (features.data < 0).any():
            raise ValueError("Multinomial naive Bayes requires nonnegative features.")
        classes, class_counts = np.unique(labels, return_counts=True)

        word_counts = np.vstack([
            np.asarray(features[labels == label].sum(axis=0)).ravel()
            for label in classes
        ])
        smoothed = word_counts + self.alpha
        self.feature_log_prob_ = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))
        self.class_log_prior_ = np.log(class_counts) - np.log(len(labels))
        self.classes_ = classes
        return self

    def log_scores(self, features):
        check_is_fitted(self, "classes_")
        features = check_array(csr_matrix(features, dtype=np.float64), accept_sparse="csr")
        if features.shape[1] != self.feature_log_prob_.shape[1]:
            raise ValueError("Prediction data must use the training vocabulary.")
        if (features.data < 0).any():
            raise ValueError("Multinomial naive Bayes requires nonnegative features.")
        return features @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, features):
        return self.classes_[self.log_scores(features).argmax(axis=1)]
