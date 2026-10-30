"""Protect against data leakage through duplicate text and shared products."""

import pandas as pd
import pytest
from hierarchical_classifier.ml.data import overlap_counts, prepare_data, split_data
from hierarchical_classifier.ml.preprocessing import combine_text


def test_same_product_and_text_stay_in_one_component():
    frame = pd.DataFrame(
        [
            {
                "productId": f"p{i}",
                "Title": f"title {i}",
                "Text": f"body {i}",
                "Cat1": "a",
                "Cat2": "b",
                "Cat3": "c",
            }
            for i in range(40)
        ]
    )
    frame.loc[1, "productId"] = "p0"
    frame.loc[2, ["Title", "Text"]] = frame.loc[1, ["Title", "Text"]].values
    frame.loc[2, "Cat3"] = "different-label"
    prepared, _ = prepare_data(frame)
    assert prepared.loc[:2, "group"].nunique() == 1
    assert all(
        not any(v.values()) for v in overlap_counts(split_data(prepared, 42)).values()
    )


def test_blank_and_duplicate_records_are_counted():
    frame = pd.DataFrame(
        [
            {"Title": f"t{i}", "Text": f"v{i}", "Cat1": "a", "Cat2": "b", "Cat3": "c"}
            for i in range(20)
        ]
    )
    frame.loc[1] = frame.loc[0]
    frame.loc[2, "Text"] = " "
    result, audit = prepare_data(frame)
    assert len(result) == 18
    assert audit["blank_rows_removed"] == audit["duplicate_rows_removed"] == 1


def test_missing_training_columns_rejected():
    with pytest.raises(ValueError, match="Missing training columns"):
        prepare_data(pd.DataFrame({"Text": ["hello"]}))


def test_preprocessing_is_deterministic_and_normalizes_unicode():
    assert combine_text("  ＣＡＴ  ", "FOOD\n good") == "cat food good"
