"""Create an xlsx to export data into an external tool.

This script is not part of the main analysis pipeline.

Requirements:
    - a_collect_image_data.py was run

Outputs:
    - xlsx with image file names and score

"""

# %%
from pathlib import Path

import pandas as pd

from depression_mapping_tools.utils import Cols

DATA_CSV = Path(__file__).parents[1] / "a_collect_image_data.csv"

PSD_SCORE_INV = "DepressionScoreInverted"
PATH_COLS = [Cols.PATH_DISCMAP_IMAGE, Cols.PATH_LESION_IMAGE, Cols.PATH_LNM_IMAGE]

# %%
# load df and drop excluded subjects
data_df = pd.read_csv(DATA_CSV)
data_df = data_df[data_df[Cols.EXCLUDED] == 0]

# %%
# fetch/modify/generate relevant cols
data_df = data_df[
    [
        Cols.PATH_DISCMAP_IMAGE,
        Cols.PATH_LESION_IMAGE,
        Cols.PATH_LNM_IMAGE,
        Cols.SUBJECT_ID,
        Cols.DEPRESSION_SCORE,
    ]
]

data_df[Cols.DEPRESSION_SCORE] = data_df[Cols.DEPRESSION_SCORE].round(3)
data_df[PSD_SCORE_INV] = -data_df[Cols.DEPRESSION_SCORE]

# transform paths to name without extension
for c in PATH_COLS:
    data_df[c] = data_df[c].apply(
        lambda p: Path(p).name.removesuffix(".nii.gz") if pd.notna(p) else pd.NA
    )

# %%
# export
outname = Path(__file__).with_suffix(".xlsx")
data_df.to_excel(outname, index=False)

# %%
