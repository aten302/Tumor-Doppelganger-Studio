<h1 align="center">Tumor Doppelgänger Studio</h1>

<p align="center">
  <i>Similarity-first interpretability: find a case’s closest look-alikes (“twins”) and explain why.</i><br/>
  <b>Educational demo only, Not a medical device.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-kNN-1F77B4?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Interpretability-Similarity--First-6C5CE7" />
  <img src="https://img.shields.io/badge/License-MIT-2EA44F" />
</p>



---

## Medical disclaimer (read first)

This project is an **educational visualization** built on a public dataset.  
It is **NOT** intended for diagnosis, treatment, screening, triage, or clinical decision-making.

**Never use this app to make medical decisions.** If  need medical support, consult qualified healthcare professionals.

---

## Why this project is different

Most ML demos do this:

> “Predict malignant vs benign.”

This project does something more *interpretable*:

> “Show me the closest look-alike cases (twins) and explain the similarity structure around a case.”

Instead of pretending the model “knows,” the app reveals **neighborhood evidence**:
- *Who does this case resemble?*
- *How consistent is the neighborhood?*
- *Which features make the case look like its twins?*
- *Is the case sitting near a boundary between groups?*
- *What minimal feature shifts would move it closer to another neighborhood?* (educational geometry, not medical advice)

---

## What the app does (high-level)

For a selected case (row):
1. **Standardizes features** to make distances meaningful across mixed units.
2. Builds a **k-nearest neighbor (kNN)** neighborhood (the “twins”).
3. Shows neighborhood composition (**how many benign vs malignant twins**).
4. Explains similarity by highlighting the **largest feature differences** (“drivers”).
5. Provides multiple interpretability views:
   - **Overview** (neighborhood composition + summary)
   - **Twin Gallery** (closest benign + malignant lists)
   - **Difference Fingerprint** (standardized deltas, if both groups exist)
   - **Minimal-Change Lab** (directional shifts toward a target group)
   - **Dataset Explorer** (full transparency: browse rows + columns)

---

## Table of contents

- [How it works](#how-it-works)
- [Screenshots & walkthrough](#screenshots--walkthrough)
- [Install & run](#install--run)
- [CLI usage](#cli-usage)
- [Project structure](#project-structure)
- [How to interpret outputs](#how-to-interpret-outputs)
- [Dataset & copyright](#dataset--copyright)
- [Limitations](#limitations--responsible-use)

---

## How it works

### 1) Feature preparation
Breast cancer datasets typically include:
- a `diagnosis` label (**B** benign, **M** malignant)
- numeric feature columns (radius_mean, texture_mean, perimeter_mean, etc.)

The app:
- loads the dataset from `data/raw/data.csv`
- selects numeric features
- standardizes them (so “area_mean” doesn’t dominate distance by scale alone)

**Why standardization matters**
Distance-based models are extremely sensitive to feature scale:
- area-related features can be orders of magnitude larger than smoothness features
- without scaling, the “largest unit features” dominate similarity even if they’re not the meaningful reason

Standardization makes distance represent **pattern similarity**, not unit mismatch.

---

### 2) Similarity via kNN neighborhoods
We model each case as a vector **x** in standardized feature space.

For a chosen case:
- find its **k nearest neighbors**
- compute distances: smaller distance = more similar overall pattern

This creates:
- a “twin list” (nearest neighbors)
- a “distance curve” (how quickly similarity fades as rank increases)

---

### 3) “Drivers” (explanation by feature deltas)
To explain why a neighbor is close to the query, we compute per-feature differences in standardized space.

The app summarizes:
- “Top drivers” = features that change the most (by absolute delta)
- not causal, but highly interpretable as geometry

Think of it as:
> “Which dimensions separate this case from its nearest look-alikes?”

---

### 4) Group contrast: benign-like vs malignant-like
If the neighborhood includes at least one benign and one malignant example:
- the app can compare “nearest benign” vs “nearest malignant”
- producing a **difference fingerprint**: the features that most separate those two reference twins

If the neighborhood is all one class (all malignant):
- the app shows a warning (because contrast needs both groups)
- that warning is itself interpretability: it indicates local homogeneity

---

### 5) Minimal-change lab (counterfactual geometry, educational)
This tool answers a **geometry** question:

> “What direction in feature space would move this case closer to the target neighborhood centroid?”

It does **NOT** say:
- “change this feature in the real world”
- “this is an intervention”
- “this is causal”

It visualizes:
- which feature dimensions dominate the shift toward benign-like or malignant-like neighborhoods.

---

## Screenshots & walkthrough

### 1) Overview, neighborhood composition & “local context”

<img width="1306" height="505" alt="Screenshot 2025-12-21 at 14-00-57 Tumor Doppelgänger Studio" src="https://github.com/user-attachments/assets/a8d5d8de-e7d5-474a-9000-5e1f66238a76" />

**What you’re seeing**

* Left sidebar “Controls”:

  * **Case row index**: which sample you are analyzing
  * **Neighbors (k)**: how many closest twins to show
  * **Top drivers**: how many explanation features to list
  * **Radar features**: how many features to display in radar plots
* Main “Overview”:

  * shows the query’s dataset label
  * counts benign vs malignant twins
  * displays malignant share (percentage of malignant neighbors)

**How to interpret**

* **Malignant share ~100%** means the case sits inside a malignant-like region of feature space.
* **Mixed neighbors** suggest boundary behavior: the case sits near regions of both types.
* A “homogeneous neighborhood” (all one label) is a strong interpretability signal: the case has many close look-alikes of that same label.

**Why this matters**
This is “context before conclusion.”
Even if someone later adds a classifier, the neighborhood evidence helps you judge:

* “Is this case supported by consistent local examples?”
* “Or is it geometrically ambiguous?”

---

### 2) Neighbor distance curve, how fast similarity decays

<img width="971" height="458" alt="Screenshot 2025-12-21 at 14-01-18 Tumor Doppelgänger Studio" src="https://github.com/user-attachments/assets/3d09d74a-817a-42e7-8e98-a4c1402e5dd8" />

**What you’re seeing**

* X-axis: neighbor rank (1 = closest)
* Y-axis: distance in standardized feature space

**How to interpret the curve**

* **Flat curve early**: many very close twins exist → strong local cluster.
* **Sharp jump**: after a small number of neighbors, similarity quickly drops → the real neighborhood might be small (k should be smaller).
* **Smooth gradual increase**: similarity decays slowly → larger k can still represent “local context.”

**How to use it to choose k**

* If the distance “knee” happens at rank 3–5, choose k around that range for meaningful neighborhood analysis.
* If there’s no knee, k=10–20 can still be reasonable for exploration.

---

### 3) Twin Gallery, closest benign and malignant look-alikes

<img width="1093" height="457" alt="Screenshot 2025-12-21 at 14-02-08 Tumor Doppelgänger Studio" src="https://github.com/user-attachments/assets/09c57581-150b-4577-892c-bb5c106efa4c" />

**What you’re seeing**

* Two lists/tables:

  * **Closest benign twins**
  * **Closest malignant twins**
* Each neighbor includes:

  * distance
  * diagnosis label
  * feature values (to inspect “what makes them similar”)

**Important behavior**
Sometimes you’ll see “No benign twins in this neighborhood.”
That’s not a bug; it means:

* within the chosen k, all closest cases are malignant (or vice versa)
* the local region is label-homogeneous

**What you can do**

* Increase k to “reach further” into the space
* Try another row index
* Use the distance curve to pick a meaningful k

---

### 4) Radar comparison, shape signature of query vs reference twins

<img width="1120" height="383" alt="Screenshot 2025-12-21 at 14-02-29 Tumor Doppelgänger Studio" src="https://github.com/user-attachments/assets/d70892cd-7508-4cc9-b86f-417d7c2269c8" />

**What you’re seeing**

* Radar charts compare:

  * the **query**
  * the nearest benign
  * the nearest malignant
* The radar is a “signature” view: it emphasizes pattern geometry.

**How to interpret**

* If the query radar shape closely matches the malignant twin, the case is geometrically malignant-like.
* If it lies between them, the case may be near a boundary.
* Radar works best when you restrict features (too many features makes radar unreadable), which is why you can select “Radar features.”

---

### 5) Difference Fingerprint, standardized contrast (if both groups exist)

<img width="1110" height="334" alt="Screenshot 2025-12-21 at 14-02-39 Tumor Doppelgänger Studio" src="https://github.com/user-attachments/assets/7d52454f-4361-4e5c-9b5e-98388adbd263" />

**What you’re seeing**

* A warning message appears if the neighborhood doesn’t include at least one benign and one malignant reference.
* This page aims to compute:

  * standardized deltas between the query and reference neighbors
  * plus a contrast between benign-like and malignant-like reference twins

**Why this is useful**
Instead of viewing single-case features in isolation, you learn:

* “Which features actually separate the nearest benign and malignant examples around this query?”
  That’s interpretability based on *local evidence*, not global averages.

**If you keep seeing the warning**

* increase k until you capture at least one neighbor of each type
* or select a case that lives nearer the boundary

---

### 6) Minimal-Change Lab, shift toward malignant-like neighborhood (educational)

<img width="1097" height="595" alt="Screenshot 2025-12-21 at 14-03-24 Tumor Doppelgänger Studio" src="https://github.com/user-attachments/assets/9bc032b6-3274-4a48-bb5e-4292a906f233" />

**What you’re seeing**

* You choose a target resemblance group (malignant-like)
* The bar chart shows directional deltas per feature

**Interpretation**

* Big deltas: features that most define the difference between the query and the target centroid
* Small deltas: features already aligned with the target group

**Critical caution**
This is **counterfactual geometry**, not causal intervention.
It does NOT imply:

* “change this biological trait”
* “this is treatment”
  It only visualizes: “these dimensions matter most for shifting similarity in this dataset.”

---

### 7) Minimal-Change Lab, shift toward benign-like neighborhood (educational)

<img width="1103" height="583" alt="Screenshot 2025-12-21 at 14-03-10 Tumor Doppelgänger Studio" src="https://github.com/user-attachments/assets/e459f656-9d3e-4d6f-9b20-cb4467860f77" />

Same tool, different target group.

**Why this view is powerful**
Comparing benign-like vs malignant-like shifts helps  see:

* whether the query is “closer” to one group than the other
* which features dominate movement toward each group

If benign-like shifts are huge but malignant-like are small:

* the case is geometrically much closer to malignant regions

---

### 8) Dataset Explorer, transparency layer (rows, columns, class counts)

<img width="1091" height="546" alt="Screenshot 2025-12-21 at 14-03-40 Tumor Doppelgänger Studio" src="https://github.com/user-attachments/assets/711374cf-0dee-4a87-bce5-dd089be7f44c" />

**What ’re seeing**

* dataset size information (rows, features)
* class distribution (benign vs malignant)
* a scrollable table view of the dataset

**Why it matters**
Interpretability without transparency can be misleading.
This page ensures  can:

* verify columns
* inspect values
* confirm preprocessing assumptions
* understand the dataset foundation behind every chart

---

## Install & run

### 1) Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the Streamlit app

```bash
streamlit run app/app.py
```

---

## CLI usage

Examples:

```bash
python -m src.cli prepare
python -m src.cli index
```

Use these when  want deterministic rebuilds of:

* cleaned dataset
* neighbor index artifacts

---

## Project structure

```text
Tumor-Doppelgänger-Studio/
├─ app/
│  └─ app.py                    # Streamlit UI (tabs, plots, controls)
├─ src/
│  ├─ cli.py                    # CLI entrypoints (prepare/index/etc.)
│  ├─ config.py                 # Paths + constants
│  ├─ data_prep.py              # Load/clean/standardize features
│  ├─ similarity.py             # kNN index + neighbor queries
│  ├─ explain.py                # Driver explanations + deltas
│  └─ utils.py                  # helpers
├─ data/
│  ├─ raw/
│  │  └─ data.csv               # Kaggle dataset copy
│  └─ processed/
│     └─ clean.csv              # processed dataset used by app
├─ models/
│  └─ twin_index.joblib         # saved neighbor index (rebuildable)
├─ requirements.txt
├─ LICENSE
└─ README.md
```

---

## How to interpret outputs

### “Neighborhood mix” is an uncertainty hint

* **All malignant neighbors**: strong malignant-like region (in this dataset’s geometry)
* **All benign neighbors**: strong benign-like region
* **Mixed neighbors**: boundary behavior (the most interesting cases for interpretability)

### Distance curve tells  whether k is meaningful

* If distance rises sharply after rank ~3, r “real neighborhood” is small.
* If distance grows gradually, k=10–20 remains locally meaningful.

### Drivers are *why the geometry looks this way*

Drivers are not “cause.”
They’re “what feature dimensions separate the query and its neighbors.”

### Minimal-change is educational counterfactual geometry

It answers:

* “what shift would move this vector toward a different neighborhood?”
  It does not imply real-world action.

---

## Dataset & copyright

**Dataset link (Kaggle):**
[https://www.kaggle.com/datasets/neurocipher/breast-cancer-dataset](https://www.kaggle.com/datasets/neurocipher/breast-cancer-dataset)

**Copyright / licensing note**

* The dataset belongs to its original authors/uploaders.
* Kaggle datasets can have specific licenses/terms-of-use on the dataset page.
* This repository uses the dataset for **educational/demo** purposes.
* If  publish/redistribute, review and comply with the dataset’s Kaggle license and terms.

---

## Limitations & responsible use

* **Not clinical**: no medical decisions.
* **Dataset-bound**: similarity is only meaningful relative to the dataset’s feature distributions.
* **Distance is a proxy**: closeness depends on chosen features + scaling choice.
* **Not causal**: deltas and minimal-change are geometry-based explanations.
