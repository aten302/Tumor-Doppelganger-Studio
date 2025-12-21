from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))  # allow `import src` when run from anywhere

from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.data_prep import clean_data
from src.explain import difference_fingerprint, minimal_shift_toward_centroid, split_twins, top_drivers
from src.similarity import TwinIndex, build_index, load_index, query_neighbors
from src.utils import format_feature_name


PLOTLY_CONFIG = {
    "displayModeBar": True,
    "responsive": True,
    "scrollZoom": True,
    "displaylogo": False,
}


@st.cache_data(show_spinner=False)
def load_clean_df() -> Tuple[pd.DataFrame, List[str]]:
    df_raw = pd.read_csv(str(config.DATA_RAW_PATH))
    prepared = clean_data(df_raw)
    return prepared.df, prepared.feature_cols


@st.cache_resource(show_spinner=False)
def get_index(df: pd.DataFrame, feature_cols: List[str]) -> TwinIndex:
    # Prefer saved index; fallback to in-memory build
    try:
        return load_index()
    except Exception:
        prepared = type("Prepared", (), {"df": df, "feature_cols": feature_cols})
        return build_index(prepared, n_neighbors=60)


def _radar(series: pd.Series, title: str) -> go.Figure:
    cols = list(series.index)
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=series.values,
            theta=[format_feature_name(c) for c in cols],
            fill="toself",
            name=title,
        )
    )
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True)),
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    return fig


def _fingerprint(drivers: pd.Series, title: str) -> go.Figure:
    dfp = pd.DataFrame({"feature": [format_feature_name(i) for i in drivers.index], "delta": drivers.values})
    fig = px.bar(dfp, x="delta", y="feature", orientation="h", title=title)
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def main() -> None:
    st.set_page_config(page_title=config.APP_TITLE, page_icon="🧬", layout="wide")

    st.title("Tumor Doppelgänger Studio")
    st.caption("Similarity-first interpretability: find a case’s closest look-alikes (twins) and explain why.")
    st.markdown(config.DISCLAIMER)

    df, feature_cols = load_clean_df()
    index = get_index(df, feature_cols)

    with st.sidebar:
        st.header("Controls")
        case_idx = st.number_input("Case row index", 0, len(df) - 1, 0, 1)
        k = st.slider("Neighbors (k)", 3, 25, 10, 1)
        topk = st.slider("Top drivers", 5, 20, 12, 1)
        radar_n = st.slider("Radar features", 6, 14, 10, 1)
        st.divider()
        st.caption("Tip: boundary cases often have mixed benign/malignant neighborhoods.")

    query = df.iloc[int(case_idx)]

    dists, idxs = query_neighbors(index, query, k=k + 1)
    # remove itself
    pairs = [(float(d), int(i)) for d, i in zip(dists, idxs) if int(i) != int(case_idx)]
    pairs = pairs[:k]

    twins = df.iloc[[i for _, i in pairs]].copy()
    twins["distance"] = [d for d, _ in pairs]

    b_count = int((twins[config.LABEL_COL] == "B").sum())
    m_count = int((twins[config.LABEL_COL] == "M").sum())
    mix_ratio = m_count / max(1, (b_count + m_count))

    tabs = st.tabs(["Overview", "Twin Gallery", "Difference Fingerprint", "Minimal-Change Lab", "Dataset Explorer"])

    # Overview
    with tabs[0]:
        st.subheader("Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Query label (dataset)", str(query[config.LABEL_COL]))
        c2.metric("Benign twins", str(b_count))
        c3.metric("Malignant twins", str(m_count))
        c4.metric("Malignant share", f"{mix_ratio:.1%}")

        st.markdown("""
**What this tool is**
- A **similarity explorer** over a public dataset.
- It shows which cases are closest to your selected case, and which features drive those similarities.

**What this tool is NOT**
- Not a diagnostic system.
- Not medical advice.
- Not a replacement for clinical evaluation.

**Neighborhood mix = uncertainty hint**
- If nearest neighbors are mixed, the case sits near a boundary region in feature-space.
        """)

        curve = twins[[config.LABEL_COL, "distance"]].copy()
        curve["rank"] = np.arange(1, len(curve) + 1)
        fig = px.line(curve, x="rank", y="distance", color=config.LABEL_COL, markers=True, title="Neighbor distance curve")
        st.plotly_chart(fig, config=PLOTLY_CONFIG, width="stretch")

    # Twin Gallery
    with tabs[1]:
        st.subheader("Twin Gallery")
        st.caption("Closest benign and malignant look-alikes (by distance).")

        groups = split_twins(twins, label_col=config.LABEL_COL)
        left, right = st.columns(2)

        with left:
            st.markdown("### Closest **Benign** twins")
            if len(groups.benign) == 0:
                st.info("No benign twins in this neighborhood. Increase k or choose another case.")
            else:
                st.dataframe(
                    groups.benign[[config.ID_COL, "distance", config.LABEL_COL] + feature_cols[:6]].sort_values("distance").head(6),
                    width="stretch",
                )

        with right:
            st.markdown("### Closest **Malignant** twins")
            if len(groups.malignant) == 0:
                st.info("No malignant twins in this neighborhood. Increase k or choose another case.")
            else:
                st.dataframe(
                    groups.malignant[[config.ID_COL, "distance", config.LABEL_COL] + feature_cols[:6]].sort_values("distance").head(6),
                    width="stretch",
                )

        st.divider()
        st.markdown("### Radar: Query vs nearest benign vs nearest malignant")
        radar_cols = feature_cols[:radar_n]

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.plotly_chart(_radar(query[radar_cols].astype(float), "Query"), config=PLOTLY_CONFIG, width="stretch")
        with rc2:
            if len(groups.benign) > 0:
                nb = groups.benign.sort_values("distance").iloc[0]
                st.plotly_chart(_radar(nb[radar_cols].astype(float), "Nearest Benign"), config=PLOTLY_CONFIG, width="stretch")
            else:
                st.info("No benign neighbor.")
        with rc3:
            if len(groups.malignant) > 0:
                nm = groups.malignant.sort_values("distance").iloc[0]
                st.plotly_chart(_radar(nm[radar_cols].astype(float), "Nearest Malignant"), config=PLOTLY_CONFIG, width="stretch")
            else:
                st.info("No malignant neighbor.")

    # Difference Fingerprint
    with tabs[2]:
        st.subheader("Difference Fingerprint")
        st.caption("Standardized deltas between the query and its nearest benign vs malignant twin.")

        groups = split_twins(twins, label_col=config.LABEL_COL)
        if len(groups.benign) == 0 or len(groups.malignant) == 0:
            st.warning("Need at least one benign and one malignant neighbor to show both fingerprints. Increase k or change case.")
        else:
            nb = groups.benign.sort_values("distance").iloc[0]
            nm = groups.malignant.sort_values("distance").iloc[0]
            stds = df[feature_cols].std(numeric_only=True)

            diff_b = difference_fingerprint(query, nb, feature_cols, zscale=stds)
            diff_m = difference_fingerprint(query, nm, feature_cols, zscale=stds)

            drv_b = top_drivers(diff_b, k=topk)
            drv_m = top_drivers(diff_m, k=topk)

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(_fingerprint(drv_b, "Query, Nearest Benign (standardized)"), config=PLOTLY_CONFIG, width="stretch")
                st.caption("Positive: query higher than benign twin. Negative: query lower.")
            with c2:
                st.plotly_chart(_fingerprint(drv_m, "Query, Nearest Malignant (standardized)"), config=PLOTLY_CONFIG, width="stretch")
                st.caption("Interpretation is geometric similarity only.")

            st.markdown("""
### How to interpret
- These bars answer: **Which measurements separate this case from its closest look-alikes?**
- Compare both sides:
  - If the query is closer to malignant on many drivers, it sits nearer malignant structure in this dataset.
  - If it is closer to benign, it sits nearer benign structure.
            """)

    # Minimal-Change Lab
    with tabs[3]:
        st.subheader("Minimal-Change Lab")
        st.caption("Counterfactual geometry: move the feature vector toward a group centroid (educational only).")

        target = st.radio("Target resemblance", ["Benign-like", "Malignant-like"], horizontal=True)
        benign_centroid = df[df[config.LABEL_COL] == "B"][feature_cols].mean()
        malignant_centroid = df[df[config.LABEL_COL] == "M"][feature_cols].mean()
        centroid = benign_centroid if target == "Benign-like" else malignant_centroid

        path = minimal_shift_toward_centroid(query, centroid, feature_cols, top_k=10)

        start = path.iloc[0][feature_cols].astype(float)
        end = path.iloc[-1][feature_cols].astype(float)
        delta = end - start
        top = delta.reindex(delta.abs().sort_values(ascending=False).index).head(12)

        df_top = pd.DataFrame({"feature": [format_feature_name(i) for i in top.index], "delta": top.values})
        fig = px.bar(df_top, x="delta", y="feature", orientation="h", title="Directional shifts toward target centroid")
        st.plotly_chart(fig, config=PLOTLY_CONFIG, width="stretch")

        st.markdown("### Path preview (alpha 0 → 1)")
        st.dataframe(path[["alpha"] + feature_cols[:10]], width="stretch")

        st.info(
            "This is a feature-space explanation tool, not a biological intervention. "
            "It shows what would need to change mathematically for the case to resemble the target group more."
        )

    # Dataset Explorer
    with tabs[4]:
        st.subheader("Dataset Explorer")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Features", f"{len(feature_cols):,}")
        c3.metric("Benign (B)", f"{(df[config.LABEL_COL] == 'B').sum():,}")
        c4.metric("Malignant (M)", f"{(df[config.LABEL_COL] == 'M').sum():,}")

        st.dataframe(df[[config.ID_COL, config.LABEL_COL] + feature_cols].head(30), width="stretch")
        st.download_button(
            "Download cleaned CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="breast_cancer_clean.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
