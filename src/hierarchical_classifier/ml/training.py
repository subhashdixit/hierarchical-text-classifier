"""Train reproducible candidates and evaluate the validation-selected model."""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml

from hierarchical_classifier.config import ROOT
from hierarchical_classifier.ml.artifacts import save_bundle, sha256
from hierarchical_classifier.ml.data import (
    LABELS,
    overlap_counts,
    prepare_data,
    split_data,
)
from hierarchical_classifier.ml.evaluation import evaluate
from hierarchical_classifier.services.taxonomy import PathClassifier

LOGGER = logging.getLogger(__name__)


def train(
    data_path: Path, config_path: Path, model_dir: Path, report_dir: Path
) -> dict:
    """Select on validation exact-path accuracy, then evaluate the test set once."""
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text())
    frame, audit = prepare_data(
        pd.read_csv(data_path, dtype=str, keep_default_na=False)
    )
    parts = split_data(frame, int(config["seed"]))
    overlaps = overlap_counts(parts)
    if any(any(counts.values()) for counts in overlaps.values()):
        raise ValueError("Data split overlap detected.")
    vectorizer = TfidfVectorizer(
        max_features=config["max_features"],
        min_df=config["min_df"],
        ngram_range=(1, config["ngram_max"]),
        sublinear_tf=True,
        dtype=np.float32,
        strip_accents="unicode",
    )
    matrices = {"train": vectorizer.fit_transform(parts["train"].combined_text)}
    for name in ["validation", "test"]:
        matrices[name] = vectorizer.transform(parts[name].combined_text)
    labels = {name: part[LABELS].to_numpy(dtype=str) for name, part in parts.items()}
    candidates = []
    best_model = None
    best_name = ""
    best_score = -1
    for spec in config["candidates"]:
        LOGGER.info("Fitting candidate %s", spec["name"])
        estimator = PathClassifier(spec["kind"], spec["regularization"])
        estimator.fit(matrices["train"], labels["train"])
        scores = evaluate(
            labels["validation"],
            estimator.predict(matrices["validation"]),
            estimator.paths_,
        )
        candidates.append({"name": spec["name"], "validation": scores})
        if scores["exact_path_accuracy"] > best_score:
            best_score, best_model, best_name = (
                scores["exact_path_accuracy"],
                estimator,
                spec["name"],
            )
    if best_model is None:
        raise ValueError("Configuration must define at least one candidate.")
    prediction_start = time.perf_counter()
    predicted = best_model.predict(matrices["test"])
    elapsed = time.perf_counter() - prediction_start
    test_report = evaluate(labels["test"], predicted, best_model.paths_)
    train_counts = parts["train"].groupby(LABELS).size()
    rare = np.array([train_counts.get(tuple(row), 0) < 5 for row in labels["test"]])
    test_report["support_groups"] = {}
    for name, mask in [
        ("under_5_training_examples", rare),
        ("at_least_5_training_examples", ~rare),
    ]:
        test_report["support_groups"][name] = {
            "rows": int(mask.sum()),
            "exact_path_accuracy": (
                float(np.mean(np.all(labels["test"][mask] == predicted[mask], axis=1)))
                if mask.any()
                else None
            ),
        }
    most_common = parts["train"].groupby(LABELS).size().idxmax()
    majority = evaluate(
        labels["test"], np.tile(most_common, (len(predicted), 1)), best_model.paths_
    )
    report = {
        "selected_candidate": best_name,
        "selection_metric": "validation exact-path accuracy",
        "candidates": candidates,
        "test": test_report,
        "majority_baseline_test": majority,
        "prediction_ms_per_record": 1000 * elapsed / len(predicted),
        "split_rows": {name: len(part) for name, part in parts.items()},
        "overlap_counts": overlaps,
        "data_audit": audit,
        "training_seconds": time.perf_counter() - started,
        "note": (
            "Single fixed product/text-group holdout; "
            "not a benchmark or calibrated confidence estimate."
        ),
    }
    taxonomy = parts["train"].groupby(LABELS).size().reset_index(name="training_rows")
    metadata = {
        "schema_version": 1,
        "model_version": "1.0.0",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": sha256(data_path),
        "config_sha256": sha256(config_path),
        "selected_candidate": best_name,
        "training_rows": len(parts["train"]),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "valid_paths": len(best_model.paths_),
        "review_score_threshold": config["review_score_threshold"],
        "score_description": (
            "Normalized path ranking score; "
            "not a calibrated probability of correctness."
        ),
    }
    save_bundle(
        {
            "vectorizer": vectorizer,
            "classifier": best_model,
            "taxonomy": taxonomy.to_dict(orient="records"),
        },
        model_dir,
        metadata,
    )
    report_dir.mkdir(exist_ok=True, parents=True)
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    taxonomy.to_csv(report_dir / "taxonomy.csv", index=False)
    pd.concat(
        [
            part[["record_id", "group"]].assign(split=name)
            for name, part in parts.items()
        ]
    ).to_csv(report_dir / "split_manifest.csv", index=False)
    comparison = pd.DataFrame(
        {"actual_" + label: labels["test"][:, i] for i, label in enumerate(LABELS)}
    )
    for i, label in enumerate(LABELS):
        comparison["predicted_" + label] = predicted[:, i]
    comparison["record_id"] = parts["test"].record_id.to_numpy()
    comparison.to_csv(report_dir / "test_predictions.csv", index=False)
    LOGGER.info(
        "Selected %s; test exact-path accuracy %.4f",
        best_name,
        test_report["exact_path_accuracy"],
    )
    return report


def main() -> None:
    """Run offline training from any working directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/data.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training.yaml")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    train(args.data, args.config, args.model_dir, args.report_dir)


if __name__ == "__main__":
    main()
