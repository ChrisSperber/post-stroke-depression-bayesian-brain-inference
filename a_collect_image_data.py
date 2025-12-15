"""Collect data and document exclusions.

Requirements: All binary lesion masks, LNM maps, structural disconnection maps (SDSM), and
corresponding CSVs are extracted into folder DATA_DIR

Output: CSV with paths to all files and depression scores.
"""

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from nibabel.nifti1 import Nifti1Image
from nibabel.orientations import aff2axcodes

from depression_mapping_tools.config import (
    DATA_DIR,
    DISCONNECTION_MAPS,
    LESION,
    LESION_NETWORK,
    MASTER_FILE_EXCEL,
    MINIMUM_AGE_INCLUSION,
    PLACEHOLDER_FILE_NOT_EXIST,
    PLACEHOLDER_MISSING_VALUE,
)
from depression_mapping_tools.utils import (
    Cols,
    find_unique_path,
    load_nifti,
)

# %%
nifti_files = [
    file
    for file in DATA_DIR.rglob("*")
    if file.is_file() and file.name.endswith(".nii") or file.name.endswith(".nii.gz")
]

ISCHAEMIC_STROKE = "Ischaemic_Stroke"
INTRACEREBRAL_HEMORRHAGE = "ICB"
TRAUMA = "Trauma"

AETIOLOGY_MAPPING = {
    1: ISCHAEMIC_STROKE,
    2: INTRACEREBRAL_HEMORRHAGE,
    3: TRAUMA,
    4: "SAH",
    5: "Other",
}
SEX_MAPPING = {0: "Female", 1: "Male"}
HANDEDNESS_MAPPING = {
    1: "right",
    2: "left",
    3: "other",  # right handed, forcibly left
    4: "other",  # left handed, forcibly right
    5: "other",  # ambidextrous
}

STROKE_AETIOLOGIES = [ISCHAEMIC_STROKE, INTRACEREBRAL_HEMORRHAGE]
MISSING_DATA_CHAR = "#"  # placeholder sign for missing values in the mastertable

# %%
# load and format demographic data
master_table_df = pd.read_excel(MASTER_FILE_EXCEL)
# drop first row with variable coding information
master_table_df = master_table_df.drop(axis=0, index=0)
master_table_df = master_table_df.reset_index(drop=True)

data = master_table_df[[Cols.SUBJECT_ID, Cols.DEPRESSION_SCORE]].copy()
data[Cols.AGE] = master_table_df["age_years"]
data[Cols.COHORT] = master_table_df["cohort"]
data[Cols.AETIOLOGY] = master_table_df["etiology"].map(AETIOLOGY_MAPPING)
data[Cols.SEX] = master_table_df["sex"].map(SEX_MAPPING)
data[Cols.HANDEDNESS] = master_table_df["handedness"].map(HANDEDNESS_MAPPING)
data[Cols.NIHSS_ON_ADMISSION] = master_table_df["NIHSSonset"]
data[Cols.DAYS_ONSET_TO_FOLLOWUP] = master_table_df["FollowUp_daysfromonset"]
data[[Cols.GDS15, Cols.GDS30, Cols.HADS]] = master_table_df[
    [Cols.GDS15, Cols.GDS30, Cols.HADS]
]
data[Cols.BDI_II] = master_table_df["BDI-II"]
data[Cols.EXCLUDED] = 0
data[Cols.EXCLUSION_REASON] = ""

# verify that no duplicate SubjectsIDs exist
if not data[Cols.SUBJECT_ID].is_unique:
    raise ValueError("Duplicate Subject IDs found!")

# replace missing-placeholder
data = data.replace(MISSING_DATA_CHAR, PLACEHOLDER_MISSING_VALUE)

# %%
# Fetch file paths and read out lesion volume
lesion_path_list = []
discmap_path_list = []
lnm_path_list = []
lesion_volume_list = []


for index, row in data.iterrows():
    sid = row[Cols.SUBJECT_ID]
    # Leipzig Subject IDs do not have fixed number count (e.g. Leipzig1 and Leipzig100 exist)
    # address by adding the extentsion, which always follows in the filename
    if "Leipzig" in sid:
        sid = f"{sid}.nii"

    lesion_path = find_unique_path(paths=nifti_files, str1=sid, str2=LESION)
    lesion_path_list.append(lesion_path)

    # read lesion volume in ml
    if lesion_path == PLACEHOLDER_FILE_NOT_EXIST:
        lesion_volume_list.append(PLACEHOLDER_MISSING_VALUE)
    else:
        img = load_nifti(lesion_path)
        img_array = img.get_fdata()
        n_voxels = np.count_nonzero(img_array)
        voxel_sizes = img.header.get_zooms()[:3]
        voxel_volume_mm3 = np.prod(voxel_sizes)
        volume_ml = n_voxels * voxel_volume_mm3 / 1000
        lesion_volume_list.append(volume_ml)

    discmap_path = find_unique_path(
        paths=nifti_files, str1=sid, str2=DISCONNECTION_MAPS
    )
    discmap_path_list.append(discmap_path)
    lnm_path = find_unique_path(paths=nifti_files, str1=sid, str2=LESION_NETWORK)
    lnm_path_list.append(lnm_path)

    if PLACEHOLDER_FILE_NOT_EXIST in (lesion_path, lnm_path, discmap_path):
        data.loc[index, Cols.EXCLUDED] = 1  # type: ignore
        data.loc[index, Cols.EXCLUSION_REASON] = "Incomplete Images"  # type: ignore

data[Cols.LESION_VOLUME] = lesion_volume_list
data[Cols.PATH_LESION_IMAGE] = lesion_path_list
data[Cols.PATH_LNM_IMAGE] = lnm_path_list
data[Cols.PATH_DISCMAP_IMAGE] = discmap_path_list


# %%
# Exclusions based on demographic/clinical information
for index, row in data.iterrows():
    if row[Cols.EXCLUDED] == 0:
        if row[Cols.AGE] < MINIMUM_AGE_INCLUSION:
            data.loc[index, Cols.EXCLUDED] = 1  # type: ignore
            data.loc[index, Cols.EXCLUSION_REASON] = "Non-adult"  # type: ignore
        elif row[Cols.AETIOLOGY] not in STROKE_AETIOLOGIES:
            data.loc[index, Cols.EXCLUDED] = 1  # type: ignore
            data.loc[index, Cols.EXCLUSION_REASON] = f"Non-stroke aetiology ({row[Cols.AETIOLOGY]})"  # type: ignore

# %%
# check image orientation according to header
example_lesion: Nifti1Image = load_nifti(data.loc[0, Cols.PATH_LESION_IMAGE])  # type: ignore
orientation = aff2axcodes(example_lesion.affine)
print("Image orientation - Lesion:", orientation)
example_lnm: Nifti1Image = load_nifti(data.loc[0, Cols.PATH_LNM_IMAGE])  # type: ignore
orientation = aff2axcodes(example_lnm.affine)
print("Image orientation - LNM:", orientation)
example_discmap: Nifti1Image = load_nifti(data.loc[0, Cols.PATH_DISCMAP_IMAGE])  # type: ignore
orientation = aff2axcodes(example_discmap.affine)
print("Image orientation - Disconnection Map:", orientation)

# %%
output_name = Path(__file__).with_suffix(".csv")
data.to_csv(output_name, index=False)

# %%
