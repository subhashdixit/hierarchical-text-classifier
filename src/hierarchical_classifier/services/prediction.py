"""Validated single/batch prediction using an approved model artifact."""

import argparse
import json
from pathlib import Path

import numpy as np

from hierarchical_classifier.config import Config
from hierarchical_classifier.ml.artifacts import load_bundle
from hierarchical_classifier.ml.preprocessing import combine_text


class ClassifierService:
    """Load once per process; never retrain or persist user inputs while serving."""

    def __init__(self, model_dir: Path):
        self.bundle, self.metadata = load_bundle(model_dir)

    @property
    def taxonomy(self) -> list[dict]:
        """Expose counts and category paths without the original review text."""
        return self.bundle["taxonomy"]

    def predict(self, title: str, text: str) -> dict:
        """Return a path, ranked alternatives and explicit review reasons."""
        if not isinstance(title, str) or not isinstance(text, str):
            raise ValueError("Title and Text must be strings.")
        if not title.strip() and not text.strip():
            raise ValueError("Enter a title or text before classifying.")
        if len(title) + len(text) > Config.MAX_TEXT_LENGTH:
            raise ValueError(
                f"Combined title and text must not exceed "
                f"{Config.MAX_TEXT_LENGTH} characters."
            )
        combined = combine_text(title, text)
        features = self.bundle["vectorizer"].transform([combined])
        classifier = self.bundle["classifier"]
        reasons = []
        if features.nnz == 0:
            return {
                "status": "unsupported",
                "categories": None,
                "path_score": None,
                "alternatives": [],
                "needs_review": True,
                "review_reasons": [
                    "No recognized vocabulary. Try descriptive English text "
                    "from the supported domains."
                ],
                "model_version": self.metadata["model_version"],
            }
        scores = classifier.path_scores(features)[0]
        indices = np.argsort(-scores, kind="stable")[:3]
        best = int(indices[0])
        if scores[best] < self.metadata["review_score_threshold"]:
            reasons.append("Low path-ranking score; review manually.")
        if len(combined.split()) < 4:
            reasons.append("Very short input; add more context.")
        if "unknown" in classifier.paths_[best]:
            reasons.append(
                "Training taxonomy contains an unknown category at this path."
            )
        return {
            "status": "predicted",
            "categories": classifier.paths_[best].tolist(),
            "path_score": float(scores[best]),
            "needs_review": bool(reasons),
            "review_reasons": reasons,
            "model_version": self.metadata["model_version"],
            "alternatives": [
                {
                    "categories": classifier.paths_[int(index)].tolist(),
                    "path_score": float(scores[index]),
                }
                for index in indices
            ],
        }


def main() -> None:
    """Classify text from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", default="")
    parser.add_argument("--text", required=True)
    parser.add_argument("--model-dir", type=Path, default=Config.MODEL_DIR)
    args = parser.parse_args()
    print(
        json.dumps(
            ClassifierService(args.model_dir).predict(args.title, args.text), indent=2
        )
    )


if __name__ == "__main__":
    main()
