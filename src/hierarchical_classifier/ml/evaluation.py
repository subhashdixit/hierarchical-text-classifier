"""Evaluate all records, including unsupported labels, without hiding errors."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from hierarchical_classifier.ml.data import LABELS


def evaluate(labels, predictions, training_paths) -> dict:
    """Calculate per-level, exact-path and unseen-label results."""
    truth = np.asarray(labels, dtype=str)
    predicted = np.asarray(predictions, dtype=str)
    known_paths = {tuple(path) for path in training_paths}
    report = {
        "rows": len(truth),
        "exact_path_accuracy": float(np.mean(np.all(truth == predicted, axis=1))),
        "invalid_path_rate": float(
            np.mean([tuple(p) not in known_paths for p in predicted])
        ),
        "unseen_true_path_rows": sum(tuple(row) not in known_paths for row in truth),
        "levels": {},
    }
    for level, name in enumerate(LABELS):
        known = set(training_paths[:, level])
        report["levels"][name] = {
            "accuracy": float(accuracy_score(truth[:, level], predicted[:, level])),
            "macro_f1": float(
                f1_score(
                    truth[:, level],
                    predicted[:, level],
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    truth[:, level],
                    predicted[:, level],
                    average="weighted",
                    zero_division=0,
                )
            ),
            "unseen_true_label_rows": sum(
                label not in known for label in truth[:, level]
            ),
        }
    categories = sorted(set(truth[:, 0]) | set(predicted[:, 0]))
    report["top_level_confusion"] = {
        "labels": categories,
        "matrix": confusion_matrix(
            truth[:, 0], predicted[:, 0], labels=categories
        ).tolist(),
    }
    return report
