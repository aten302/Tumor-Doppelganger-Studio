from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import config
from .data_prep import PreparedData, clean_data, load_raw


@dataclass
class TwinIndex:
    feature_cols: List[str]
    scaler: StandardScaler
    nn: NearestNeighbors
    X_scaled: np.ndarray
    ids: np.ndarray
    labels: np.ndarray


def build_index(prepared: PreparedData, n_neighbors: int = 50) -> TwinIndex:
    df = prepared.df
    cols = prepared.feature_cols

    X = df[cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(df)), metric="euclidean")
    nn.fit(Xs)

    return TwinIndex(
        feature_cols=cols,
        scaler=scaler,
        nn=nn,
        X_scaled=Xs,
        ids=df[config.ID_COL].to_numpy(),
        labels=df[config.LABEL_COL].to_numpy(),
    )


def save_index(index: TwinIndex, path: str | None = None) -> str:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = str(path or config.INDEX_PATH)
    joblib.dump(index, out)
    return out


def load_index(path: str | None = None) -> TwinIndex:
    p = str(path or config.INDEX_PATH)
    return joblib.load(p)


def build_and_save_index(n_neighbors: int = 50) -> str:
    try:
        df = pd.read_csv(str(config.DATA_PROCESSED_PATH))
    except Exception:
        df = load_raw()
    prepared = clean_data(df)
    idx = build_index(prepared, n_neighbors=n_neighbors)
    return save_index(idx)


def query_neighbors(index: TwinIndex, x_row: pd.Series, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    x = x_row[index.feature_cols].to_numpy(dtype=float).reshape(1, -1)
    xs = index.scaler.transform(x)
    dists, idxs = index.nn.kneighbors(xs, n_neighbors=min(k, len(index.ids)))
    return dists[0], idxs[0]
