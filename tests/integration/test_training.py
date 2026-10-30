"""Exercise a full offline train/save/load run on a small synthetic dataset."""

from pathlib import Path
import pandas as pd
from hierarchical_classifier.ml.training import train
from hierarchical_classifier.services.prediction import ClassifierService


def test_training_pipeline_produces_servable_artifacts(tmp_path):
    rows = []
    for index in range(60):
        dog = index % 2 == 0
        rows.append(
            {
                "Title": f"sample {index}",
                "Text": (
                    f"dog rubber chew toy {index}"
                    if dog
                    else f"vitamin nutrition tablet {index}"
                ),
                "productId": str(index),
                "Cat1": "pet" if dog else "health",
                "Cat2": "dog" if dog else "nutrition",
                "Cat3": "toy" if dog else "vitamin",
            }
        )
    data_path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)
    config = tmp_path / "config.yaml"
    config.write_text(
        "seed: 42\nmax_features: 100\nmin_df: 1\nngram_max: 1\nreview_score_threshold: 0.45\ncandidates:\n  - {name: tiny_nb, kind: joint_nb, regularization: 1.0}\n"
    )
    models, reports = tmp_path / "models", tmp_path / "reports"
    result = train(data_path, config, models, reports)
    assert result["selected_candidate"] == "tiny_nb"
    assert (reports / "test_predictions.csv").exists()
    assert ClassifierService(Path(models)).predict("dog", "rubber chew toy")[
        "categories"
    ]
