"""Constrain predictions to complete paths observed in training."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


class PathClassifier(ClassifierMixin, BaseEstimator):
    """Compare joint-path NB with constrained independent level classifiers.

    Level classifiers share TF-IDF features. No parent prediction is used as a
    training feature. Scores combine level probabilities along allowed paths;
    normalized path scores are rankings, not calibrated probabilities.
    """

    def __init__(self, kind: str = "levels", regularization: float = 1.0):
        self.kind = kind
        self.regularization = regularization
        self.paths_ = None
        self.estimators_ = []

    def fit(self, features, labels):
        """Fit on training labels only."""
        labels = np.asarray(labels, dtype=str)
        self.paths_, encoded = np.unique(labels, axis=0, return_inverse=True)
        if self.kind == "joint_nb":
            self.estimators_ = [MultinomialNB(alpha=0.1).fit(features, encoded)]
        else:
            self.estimators_ = [
                LogisticRegression(
                    C=self.regularization, solver="lbfgs", max_iter=600
                ).fit(features, labels[:, level])
                for level in range(3)
            ]
        return self

    def path_scores(self, features) -> np.ndarray:
        """Return normalized scores over valid paths, in stored path order."""
        if self.kind == "joint_nb":
            return self.estimators_[0].predict_proba(features)
        log_scores = np.zeros((features.shape[0], len(self.paths_)))
        for level, estimator in enumerate(self.estimators_):
            probabilities = estimator.predict_proba(features)
            index = {label: idx for idx, label in enumerate(estimator.classes_)}
            selected = [index[path[level]] for path in self.paths_]
            log_scores += np.log(np.clip(probabilities[:, selected], 1e-12, 1))
        log_scores -= log_scores.max(axis=1, keepdims=True)
        scores = np.exp(log_scores)
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, features) -> np.ndarray:
        """Predict complete category paths."""
        return self.paths_[self.path_scores(features).argmax(axis=1)]
