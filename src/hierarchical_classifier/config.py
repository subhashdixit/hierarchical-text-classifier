"""Configuration shared by the command line and web application."""

import os
from pathlib import Path

ROOT = Path(os.getenv("APP_ROOT", str(Path(__file__).resolve().parents[2])))


class Config:
    """Safe defaults for a public demo without persistent user storage."""

    MODEL_DIR = Path(os.getenv("MODEL_DIR", str(ROOT / "models")))
    REPORT_DIR = ROOT / "reports"
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024
    MAX_BATCH_ROWS = 100
    MAX_TEXT_LENGTH = 10000
    JSON_SORT_KEYS = False
    TESTING = False
