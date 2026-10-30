"""Validate data and isolate products and repeated text across splits."""

import hashlib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from hierarchical_classifier.ml.preprocessing import combine_text

LABELS = ["Cat1", "Cat2", "Cat3"]


def prepare_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Connect overlapping products/text into groups, then remove exact repeats.

    Connected components stop the same product OR normalized text appearing in
    different splits. User IDs and timestamps are not used as features.
    """
    required = ["Title", "Text", *LABELS]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing training columns: {', '.join(missing)}")
    data = frame.copy().fillna("")
    original_count = len(data)
    for column in required:
        data[column] = data[column].astype(str).str.strip()
    valid = data[required].ne("").all(axis=1)
    data = data.loc[valid].copy()
    dropped_blank = original_count - len(data)
    data["combined_text"] = [combine_text(a, b) for a, b in zip(data.Title, data.Text)]
    before_dedup = len(data)
    data = data.drop_duplicates(["combined_text", *LABELS]).reset_index(drop=True)
    if len(data) < 10:
        raise ValueError("At least 10 usable training records are required.")
    parents = list(range(len(data)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in data.iterrows():
        keys = [("text", row["combined_text"])]
        product = str(row.get("productId", "")).strip()
        if product:
            keys.append(("product", product))
        for key in keys:
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    data["group"] = [find(index) for index in range(len(data))]
    data["record_id"] = [
        hashlib.sha256(v.encode()).hexdigest() for v in data.combined_text
    ]
    audit = {
        "original_rows": original_count,
        "blank_rows_removed": dropped_blank,
        "duplicate_rows_removed": before_dedup - len(data),
        "usable_rows": len(data),
        "groups": int(data.group.nunique()),
        "classes": {label: int(data[label].nunique()) for label in LABELS},
        "rare_leaf_labels_under_5_rows": int((data.Cat3.value_counts() < 5).sum()),
    }
    return data, audit


def split_data(data: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Produce approximately 60/20/20 group-separated train/validation/test data."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    rest_idx, test_idx = next(splitter.split(data, groups=data.group))
    rest = data.iloc[rest_idx]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    train_idx, validation_idx = next(splitter.split(rest, groups=rest.group))
    parts = {
        "train": rest.iloc[train_idx].copy(),
        "validation": rest.iloc[validation_idx].copy(),
        "test": data.iloc[test_idx].copy(),
    }
    if any(part.empty for part in parts.values()):
        raise ValueError("Too few independent groups for three nonempty splits.")
    return parts


def overlap_counts(parts: dict[str, pd.DataFrame]) -> dict:
    """Report explicit cross-split group and normalized-text overlap counts."""
    result = {}
    for left, right in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        result[f"{left}_{right}"] = {
            "groups": len(set(parts[left].group) & set(parts[right].group)),
            "texts": len(set(parts[left].record_id) & set(parts[right].record_id)),
        }
    return result
