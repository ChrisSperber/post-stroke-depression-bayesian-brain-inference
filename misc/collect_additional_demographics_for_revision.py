"""Additional demographics collected for manuscript revision.

- creates figure with distribution of depressive symptom scores for each measure

"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from depression_mapping_tools.config import PLACEHOLDER_MISSING_VALUE
from depression_mapping_tools.utils import (
    DEPRESSION_GT_CUTOFFS,
    Cols,
)

COHORT = Cols.COHORT
DEPRESSION_BINARY = "DepressionBinary"
VARIABLE = "Variable"
STAT = "Stat"
VALUE = "Value"

# Cohort Names
IOWA = "Iowa"
KOREA = "Korea"
LEIPZIG = "Leipzig"
BORDEAUX = "Bordeaux"
# Korean sub-cohorts
HALLYM = "Hallym"
BUNDANG = "Bundang"

cohorts = [IOWA, KOREA, LEIPZIG, BORDEAUX]

DEPRESSION_MEASURE_MAP = {
    IOWA: [Cols.BDI_II],
    KOREA: [Cols.GDS15, Cols.GDS30],
    LEIPZIG: [Cols.HADS],
    BORDEAUX: [Cols.HADS],
}

DEPRESSION_SCORE_COLS = [Cols.GDS15, Cols.GDS30, Cols.BDI_II, Cols.HADS]

# %%
data = pd.read_csv(Path(__file__).parents[1] / "a_collect_image_data.csv")
data = data[data[Cols.EXCLUDED] == 0]

# derive binary depression classification based on cutoffs
depression_classification_list = []
for _, row in data.iterrows():
    mask = row[DEPRESSION_SCORE_COLS] != PLACEHOLDER_MISSING_VALUE
    col_name = row[DEPRESSION_SCORE_COLS][mask].index[0]
    depression_value = int(row[DEPRESSION_SCORE_COLS][mask].iloc[0])

    relevant_cutoff = DEPRESSION_GT_CUTOFFS.get(col_name)
    if relevant_cutoff:
        depression_classification_list.append(int(depression_value > relevant_cutoff))
    else:
        raise ValueError("Relevant depression cutoff was not derived!")
data[DEPRESSION_BINARY] = depression_classification_list

# %%
MEASURE_LABELS = {
    Cols.GDS15: "GDS-15",
    Cols.GDS30: "GDS-30",
    Cols.BDI_II: "BDI-II",
    Cols.HADS: "HADS",
}

rng = np.random.default_rng(42)

# Each measure occupies one column:
# upper row = jittered observations
# lower row = boxplot
fig, axes = plt.subplots(
    nrows=2,
    ncols=len(DEPRESSION_SCORE_COLS),
    figsize=(12, 4.5),
    sharex="col",
    gridspec_kw={
        "height_ratios": [4, 1],
        "hspace": 0.02,
        "wspace": 0.25,
    },
)

for col_index, measure in enumerate(DEPRESSION_SCORE_COLS):
    scatter_ax = axes[0, col_index]
    box_ax = axes[1, col_index]

    scores = pd.to_numeric(data[measure], errors="coerce")

    # Optional safeguard if the missing-value placeholder is numeric
    if isinstance(PLACEHOLDER_MISSING_VALUE, (int, float)):
        scores = scores[scores != PLACEHOLDER_MISSING_VALUE]

    scores = scores.dropna()

    # Jitter only along the y-axis
    jitter = rng.uniform(-0.25, 0.25, size=len(scores))

    scatter_ax.scatter(
        scores,
        jitter,
        s=18,
        alpha=0.45,
        linewidths=0,
    )

    scatter_ax.set_title(
        f"{MEASURE_LABELS[measure]}\n$n$ = {len(scores)}",
        fontsize=11,
    )
    scatter_ax.set_yticks([])
    scatter_ax.set_ylim(-0.4, 0.4)

    box_ax.boxplot(
        scores,
        vert=False,
        positions=[0],
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        boxprops={
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 1.0,
        },
        medianprops={
            "color": "black",
            "linewidth": 1.5,
        },
        whiskerprops={
            "color": "black",
            "linewidth": 1.0,
        },
        capprops={
            "color": "black",
            "linewidth": 1.0,
        },
    )

    box_ax.set_yticks([])
    box_ax.set_ylim(-0.7, 0.7)
    box_ax.set_xlabel("Score")
    box_ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Remove unnecessary borders
    scatter_ax.spines["left"].set_visible(False)
    scatter_ax.spines["right"].set_visible(False)
    scatter_ax.spines["top"].set_visible(False)
    scatter_ax.spines["bottom"].set_visible(False)

    box_ax.spines["left"].set_visible(False)
    box_ax.spines["right"].set_visible(False)
    box_ax.spines["top"].set_visible(False)

fig.suptitle(
    "Distribution of depression scores by measure",
    fontsize=13,
    y=1.02,
)

out_path = Path(__file__).with_name("depression_score_distributions.png")
fig.savefig(
    out_path,
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
