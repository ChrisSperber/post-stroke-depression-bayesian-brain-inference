"""Collect data and document inclusions.

Requirements: All binary lesion masks, LNM maps, and corresponding CSVs are extracted into a folder

Output: CSV with paths to all files and depression scores.
"""

# %%
from pathlib import Path

import pandas as pd

from utils.utils import PLACEHOLDER_FILE_NOT_EXIST, find_unique_path

SUBJECT_ID = "SubjectID"
DEPRESSION_SCORE = "DepressionZScore"
EXCLUDED = "Excluded"
EXCLUSION_REASON = "ExclusionReason"
PATH_LESION_IMAGE = "PathLesionImage"
PATH_LNM_IMAGE = "PathLNMImage"
PATH_DISCMAP_IMAGE = "PathDiscMapImage"

DATA_DIR = Path(r"D:\Neuro\Projekt_Depression_BLDI\Data")

# %%
files = [file for file in DATA_DIR.rglob("*") if file.is_file()]

# %%
# find all unique subjects in the excel files
excel_files = [f for f in files if f.suffix == ".xlsx"]
df_list = []

for file in excel_files:
    df_full_excel = pd.read_excel(
        file, skiprows=1, header=None
    )  # skip the inconsistent headers

    # some excel files were likely appended with duplicated header -> remove
    ids_to_drop = ["fname", "lesionNetwork"]
    df_full_excel = df_full_excel[~df_full_excel.iloc[:, 0].isin(ids_to_drop)]
    df_list.append(df_full_excel)

df_full_excel = pd.concat(df_list, ignore_index=True)
df_full_excel.columns = [SUBJECT_ID, DEPRESSION_SCORE]

# extract the ID from the first column
df_full_excel[SUBJECT_ID] = df_full_excel[SUBJECT_ID].str.split("\\\\").str[-1]
df_full_excel[SUBJECT_ID] = df_full_excel[SUBJECT_ID].str.replace(
    ".nii", "", regex=False
)
df_full_excel[SUBJECT_ID] = df_full_excel[SUBJECT_ID].str.replace(
    ".gz", "", regex=False
)
df_full_excel[SUBJECT_ID] = df_full_excel[SUBJECT_ID].str.replace(
    "mean_roi_", "", regex=False
)

# verify that the depressions scores are equivalent across excel files
duplicates_with_different_values = df_full_excel.groupby(SUBJECT_ID)[
    DEPRESSION_SCORE
].nunique()
inconsistent_ids = duplicates_with_different_values[
    duplicates_with_different_values > 1
]

if not inconsistent_ids.empty:
    print("Inconsistent SubjectIDs:")
    print(
        df_full_excel[
            df_full_excel[df_full_excel.columns[0]].isin(inconsistent_ids.index)
        ]
    )
else:
    print("No inconsistent SubjectIDs")

data = df_full_excel.drop_duplicates(subset=SUBJECT_ID, keep="first")

# %%
# Fetch file paths and document exclusions
# Exclusion of subjects with known issues
data = df_full_excel.drop_duplicates(subset=SUBJECT_ID, keep="first")
data[EXCLUDED] = 0
data[EXCLUSION_REASON] = ""

# fetch paths to images
for index, row in data.iterrows():
    if row[EXCLUDED] == 0:
        file_id = row[SUBJECT_ID] + ".nii"
        lesion_path = find_unique_path(paths=files, str1=file_id, str2="Lesions")
        data.loc[index, PATH_LESION_IMAGE] = lesion_path
        lnm_path = find_unique_path(paths=files, str1=file_id, str2="LNM")
        data.loc[index, PATH_LNM_IMAGE] = lnm_path
        discmap_path = find_unique_path(paths=files, str1=file_id, str2="DiscMaps")
        data.loc[index, PATH_DISCMAP_IMAGE] = discmap_path

        if PLACEHOLDER_FILE_NOT_EXIST in (lesion_path, lnm_path, discmap_path):
            data.loc[index, EXCLUDED] = 1
            data.loc[index, EXCLUSION_REASON] = "Incomplete Images"


# %%
output_name = Path(__file__).with_suffix(".csv")
data.to_csv(output_name, index=False)

# %%
