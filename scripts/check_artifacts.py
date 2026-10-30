"""Validate artifact integrity, split overlap and report consistency."""

import json
from hierarchical_classifier.config import ROOT
from hierarchical_classifier.ml.artifacts import load_bundle, sha256
from hierarchical_classifier.services.prediction import ClassifierService

bundle, metadata = load_bundle(ROOT / "models")
metrics = json.loads((ROOT / "reports/metrics.json").read_text())
assert metadata["dataset_sha256"] == sha256(ROOT / "data/raw/data.csv")
assert metadata["config_sha256"] == sha256(ROOT / "configs/training.yaml")
assert all(not any(v.values()) for v in metrics["overlap_counts"].values())
assert metrics["test"]["invalid_path_rate"] == 0
# Fixed-dataset regression floors, not a promise of future production accuracy.
assert metrics["test"]["exact_path_accuracy"] >= 0.40
assert metrics["test"]["levels"]["Cat1"]["accuracy"] >= 0.75
assert (
    metrics["test"]["exact_path_accuracy"]
    > metrics["majority_baseline_test"]["exact_path_accuracy"]
)
assert len(bundle["taxonomy"]) == metadata["valid_paths"]
service = ClassifierService(ROOT / "models")
assert service.predict("Dog toy", "A rubber chew toy for my dog")["categories"]
print("Approved artifacts, provenance, split overlap and inference verified.")
