"""One deterministic text transformation for training and serving."""

import re
import unicodedata


def clean_text(value: str) -> str:
    """Normalize Unicode, case and whitespace without network downloads."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def combine_text(title: str, text: str) -> str:
    """Keep title and body in the same order at training and prediction."""
    return clean_text(f"{title} {text}")
