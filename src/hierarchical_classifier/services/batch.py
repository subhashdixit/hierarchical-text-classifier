"""Parse bounded CSV uploads, preserving per-row errors and preventing formulas."""

import io
import pandas as pd

from hierarchical_classifier.config import Config
from hierarchical_classifier.services.prediction import ClassifierService


def classify_csv(content: bytes, service: ClassifierService) -> list[dict]:
    """Require Title/Text columns; retain good rows when a row fails validation."""
    if not content or len(content) > Config.MAX_CONTENT_LENGTH:
        raise ValueError("Upload a nonempty UTF-8 CSV no larger than 2 MB.")
    try:
        frame = pd.read_csv(
            io.BytesIO(content),
            dtype=str,
            keep_default_na=False,
            nrows=Config.MAX_BATCH_ROWS + 1,
            encoding="utf-8-sig",
        )
    except (ValueError, UnicodeError) as exc:
        raise ValueError(
            "Could not read CSV. Use UTF-8 with Title and Text columns."
        ) from exc
    if not {"Title", "Text"}.issubset(frame.columns):
        raise ValueError("CSV must include the exact columns Title and Text.")
    if frame.empty or len(frame) > Config.MAX_BATCH_ROWS:
        raise ValueError(
            f"CSV must contain between 1 and {Config.MAX_BATCH_ROWS} rows."
        )
    results = []
    for number, row in enumerate(frame.to_dict(orient="records"), start=1):
        try:
            prediction = service.predict(row["Title"], row["Text"])
            categories = prediction["categories"] or ["", "", ""]
            result = {
                "row": number,
                "Title": row["Title"],
                "Text": row["Text"],
                "Cat1": categories[0],
                "Cat2": categories[1],
                "Cat3": categories[2],
                "path_score": prediction["path_score"],
                "status": prediction["status"],
                "needs_review": prediction["needs_review"],
                "message": " ".join(prediction["review_reasons"]),
            }
        except ValueError as exc:
            result = {
                "row": number,
                "Title": row["Title"],
                "Text": row["Text"],
                "status": "error",
                "needs_review": True,
                "message": str(exc),
            }
        results.append(result)
    return results


def results_csv(results: list[dict]) -> str:
    """Escape spreadsheet formula prefixes in downloaded text fields."""
    frame = pd.DataFrame(results).fillna("")

    def safe_cell(value):
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + value
        return value

    for column in frame:
        frame[column] = frame[column].map(safe_cell)
    return frame.to_csv(index=False)
