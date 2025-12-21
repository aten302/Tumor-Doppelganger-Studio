from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_PATH = REPO_ROOT / "data" / "raw" / "data.csv"
DATA_PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "clean.csv"

MODEL_DIR = REPO_ROOT / "models"
INDEX_PATH = MODEL_DIR / "twin_index.joblib"

ID_COL = "id"
LABEL_COL = "diagnosis"  # 'M' or 'B' in the dataset

APP_TITLE = "Tumor Doppelgänger Studio"
DISCLAIMER = (
    "⚠️ **Educational demo only. Not a medical device.** "
    "This app explores similarity between samples in a public dataset. "
    "Do **not** use it for diagnosis or treatment decisions."
)
