from __future__ import annotations

import re
from typing import Iterable, List

import numpy as np
import pandas as pd


def pick_feature_columns(df: pd.DataFrame, exclude: Iterable[str]) -> List[str]:
    exclude_set = set(exclude)
    cols: List[str] = []
    for c in df.columns:
        if c in exclude_set:
            continue
        if str(c).lower().startswith("unnamed"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def topk_abs(series: pd.Series, k: int) -> pd.Series:
    return series.reindex(series.abs().sort_values(ascending=False).index).head(k)


def format_feature_name(name: str) -> str:
    return re.sub(r"_", " ", name).replace("  ", " ").strip().title()
