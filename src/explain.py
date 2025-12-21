from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .utils import topk_abs


@dataclass
class TwinGroup:
    benign: pd.DataFrame
    malignant: pd.DataFrame


def split_twins(df_twins: pd.DataFrame, label_col: str) -> TwinGroup:
    return TwinGroup(
        benign=df_twins[df_twins[label_col] == "B"].copy(),
        malignant=df_twins[df_twins[label_col] == "M"].copy(),
    )


def difference_fingerprint(
    query: pd.Series,
    ref: pd.Series,
    feature_cols: List[str],
    zscale: pd.Series | None = None,
) -> pd.Series:
    q = query[feature_cols].astype(float)
    r = ref[feature_cols].astype(float)
    diff = q - r
    if zscale is not None:
        s = zscale[feature_cols].replace(0, np.nan)
        diff = (diff / s).fillna(0.0)
    return diff


def top_drivers(diff: pd.Series, k: int = 12) -> pd.Series:
    return topk_abs(diff, k=k)


def minimal_shift_toward_centroid(
    query: pd.Series,
    centroid: pd.Series,
    feature_cols: List[str],
    top_k: int = 10,
) -> pd.DataFrame:
    q = query[feature_cols].astype(float)
    c = centroid[feature_cols].astype(float)
    delta = c - q

    key = topk_abs(delta, k=min(top_k, len(feature_cols))).index.tolist()
    alphas = np.linspace(0.0, 1.0, 6)

    rows = []
    for a in alphas:
        r = q.copy()
        r[key] = q[key] + a * delta[key]
        rows.append(r)

    out = pd.DataFrame(rows)
    out.insert(0, "alpha", alphas)
    return out
