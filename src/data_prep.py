from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from . import config
from .utils import pick_feature_columns


@dataclass
class PreparedData:
    df: pd.DataFrame
    feature_cols: List[str]


def load_raw(path: str | None = None) -> pd.DataFrame:
    path = str(path or config.DATA_RAW_PATH)
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> PreparedData:
    d = df.copy()

    # Drop unnamed empty columns
    unnamed = [c for c in d.columns if str(c).lower().startswith("unnamed")]
    if unnamed:
        d = d.drop(columns=unnamed)

    if config.LABEL_COL not in d.columns:
        raise ValueError(f"Missing label column '{config.LABEL_COL}'")

    d[config.LABEL_COL] = d[config.LABEL_COL].astype(str).str.strip().str.upper()
    d = d[d[config.LABEL_COL].isin(["B", "M"])].copy()

    if config.ID_COL not in d.columns:
        d[config.ID_COL] = np.arange(len(d), dtype=int)

    feature_cols = pick_feature_columns(d, exclude=[config.ID_COL, config.LABEL_COL])

    # Coerce numeric + median impute
    for c in feature_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        med = float(np.nanmedian(d[c].values))
        d[c] = d[c].fillna(med)

    return PreparedData(df=d, feature_cols=feature_cols)


def prepare_and_save(raw_path: str | None = None, processed_path: str | None = None) -> PreparedData:
    df_raw = load_raw(raw_path)
    prepared = clean_data(df_raw)

    out_path = str(processed_path or config.DATA_PROCESSED_PATH)
    config.DATA_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    prepared.df.to_csv(out_path, index=False)
    return prepared
