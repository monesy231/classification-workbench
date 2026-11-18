"""Regression checks for the from-scratch classifiers and binary scores."""

import unittest

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB

from classification_workbench.bank import binary_scores, discretize
from classification_workbench.id3 import ID3, entropy, information_gain
from classification_workbench.naive_bayes import SparseMultinomialNB


class ID3Tests(unittest.TestCase):
    def test_information_gain_and_unseen_category(self):
        features = np.array([[0], [0], [1], [1], [1]])
        labels = np.array([0, 0, 1, 1, 1])
        tree = ID3().fit(features, labels)
        self.assertAlmostEqual(information_gain(features[:, 0], labels), entropy(labels))
        np.testing.assert_array_equal(tree.predict([[0], [1], [9]]), [0, 1, 1])
        stump = ID3(max_depth=0).fit(features, labels)
        np.testing.assert_array_equal(stump.predict(features), [1] * 5)

    def test_income_bin_boundary(self):
        features = np.zeros((4, 7))
        features[:, -1] = [9_999, 10_000, 10_001, 40_001]
        np.testing.assert_array_equal(discretize(features)[:, -1], [0, 0, 1, 5])

    def test_binary_metrics_match_sklearn(self):
        actual, predicted = [0, 0, 1, 1], [0, 1, 1, 1]
        scores = binary_scores(actual, predicted)
        self.assertEqual(scores["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(scores["f1"], f1_score(actual, predicted))


class NaiveBayesTests(unittest.TestCase):
    def test_sparse_predictions_and_log_probabilities_match_sklearn(self):
        features = csr_matrix([[2, 0, 1], [0, 3, 2], [4, 0, 0], [0, 2, 1], [1, 0, 1]])
        labels = ["first", "second", "first", "second", "first"]
        probes = csr_matrix([[2, 1, 0], [0, 5, 1], [0, 0, 0]])
        for alpha in (0.05, 1.0, 10.0):
            with self.subTest(alpha=alpha):
                scratch = SparseMultinomialNB(alpha).fit(features, labels)
                reference = MultinomialNB(alpha=alpha).fit(features, labels)
                np.testing.assert_allclose(scratch.feature_log_prob_, reference.feature_log_prob_)
                np.testing.assert_allclose(scratch.class_log_prior_, reference.class_log_prior_)
                np.testing.assert_array_equal(scratch.predict(probes), reference.predict(probes))
