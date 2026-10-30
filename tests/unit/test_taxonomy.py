"""Constrained decoding must never invent an incompatible parent-child path."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from hierarchical_classifier.services.taxonomy import PathClassifier
from hierarchical_classifier.ml.evaluation import evaluate


@pytest.mark.parametrize("kind", ["levels", "joint_nb"])
def test_predictions_are_allowed_complete_paths(kind):
    features = csr_matrix([[3, 0], [4, 0], [0, 3], [0, 4]])
    labels = np.array(
        [
            ["pet", "dog", "toy"],
            ["pet", "dog", "toy"],
            ["beauty", "skin", "cream"],
            ["beauty", "skin", "cream"],
        ]
    )
    model = PathClassifier(kind).fit(features, labels)
    predicted = model.predict(features)
    assert {tuple(p) for p in predicted} <= {tuple(p) for p in labels}
    assert np.allclose(model.path_scores(features).sum(axis=1), 1)


def test_unseen_ground_truth_is_not_excluded_from_evaluation():
    paths = np.array([["a", "b", "c"]])
    metrics = evaluate([["a", "b", "new"]], paths, paths)
    assert metrics["rows"] == 1
    assert metrics["exact_path_accuracy"] == 0
    assert metrics["unseen_true_path_rows"] == 1
