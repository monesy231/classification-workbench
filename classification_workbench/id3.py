"""A small ID3 classifier for discrete, integer-valued features."""

from dataclasses import dataclass, field

import numpy as np


def entropy(labels):
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def information_gain(values, labels):
    remaining_entropy = 0.0
    for value in np.unique(values):
        subset = labels[values == value]
        remaining_entropy += (len(subset) / len(labels)) * entropy(subset)
    return entropy(labels) - remaining_entropy


@dataclass
class Node:
    majority: int
    feature: int | None = None
    children: dict[int, "Node"] = field(default_factory=dict)


class ID3:
    def __init__(self, max_depth=None):
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be nonnegative.")
        self.max_depth = max_depth

    def fit(self, features, labels):
        features = np.asarray(features)
        labels = np.asarray(labels)
        if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels) or not len(labels):
            raise ValueError("ID3 needs a nonempty 2-D feature matrix and matching labels.")
        if not np.issubdtype(features.dtype, np.integer) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("ID3 expects discrete integer feature values and labels.")
        self.root_ = self._grow(features, labels, tuple(range(features.shape[1])), depth=0)
        self.n_features_ = features.shape[1]
        return self

    def _grow(self, features, labels, available, depth):
        classes, counts = np.unique(labels, return_counts=True)
        node = Node(majority=int(classes[counts.argmax()]))
        if len(classes) == 1 or not available or depth == self.max_depth:
            return node

        node.feature = max(available, key=lambda column: information_gain(features[:, column], labels))
        remaining = tuple(column for column in available if column != node.feature)
        values = features[:, node.feature]
        for value in np.unique(values):
            subset = values == value
            node.children[int(value)] = self._grow(features[subset], labels[subset], remaining, depth + 1)
        return node

    def _predict_one(self, row):
        node = self.root_
        while node.feature is not None:
            child = node.children.get(row[node.feature])
            if child is None:
                break
            node = child
        return node.majority

    def predict(self, features):
        if not hasattr(self, "root_"):
            raise ValueError("Fit ID3 before calling predict.")
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] != self.n_features_:
            raise ValueError("Prediction data must use the training feature columns.")
        return np.array([self._predict_one(row) for row in features], dtype=int)
