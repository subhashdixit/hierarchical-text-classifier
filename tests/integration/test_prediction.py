"""Exercise real artifacts, batch handling and integrity failures."""

import shutil
import pytest
from hierarchical_classifier.config import ROOT
from hierarchical_classifier.ml.artifacts import load_bundle
from hierarchical_classifier.services.batch import classify_csv, results_csv


def test_real_prediction_and_taxonomy(app):
    service = app.extensions["classifier"]
    result = service.predict(
        "Dog toy", "My dog enjoys chewing this rubber toy every day"
    )
    assert len(result["categories"]) == 3
    assert tuple(result["categories"]) in {
        tuple(r[k] for k in ["Cat1", "Cat2", "Cat3"]) for r in service.taxonomy
    }
    assert 0 <= result["path_score"] <= 1


def test_unsupported_vocabulary_does_not_claim_a_prediction(app):
    result = app.extensions["classifier"].predict("", "qzxvplm qzxvplm")
    assert result["status"] == "unsupported"
    assert result["categories"] is None


def test_batch_preserves_row_errors(app):
    rows = classify_csv(
        b"Title,Text\nDog toy,A rubber toy for my dog\n,\n",
        app.extensions["classifier"],
    )
    assert len(rows) == 2
    assert rows[0]["status"] == "predicted"
    assert rows[1]["status"] == "error"


@pytest.mark.parametrize(
    "content",
    [b"", b"Wrong,Headers\na,b", b"Title,Text\n" + b"a,b\n" * 101, b"\xff\xfe"],
)
def test_invalid_batch_files_rejected(content, app):
    with pytest.raises(ValueError):
        classify_csv(content, app.extensions["classifier"])


def test_spreadsheet_formula_prefixes_are_escaped():
    output = results_csv([{"Title": "=1+1", "Text": " @SUM(A1)", "path_score": 0.3}])
    assert "'=1+1" in output
    assert "' @SUM(A1)" in output


def test_changed_model_rejected_before_deserialization(tmp_path):
    shutil.copy(ROOT / "models/metadata.json", tmp_path / "metadata.json")
    (tmp_path / "classifier.joblib").write_bytes(b"not a trusted artifact")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path)
