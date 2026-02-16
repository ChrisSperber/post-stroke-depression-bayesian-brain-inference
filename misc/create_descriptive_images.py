"""Create Nifti Images of lesion overlap etc. for manuscript figures.

This script is not part of the main analysis pipeline.

Requirements:
    - a_collect_image_data.py was run

Outputs:
    - lesion overlaps image, disconnection overlap (at binary threshold), and mean lnm map are
        generated at misc/descriptive_images

"""

# %%
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.nifti1 import Nifti1Image

from depression_mapping_tools.config import TRAUMA_EXCLUSION_COMMENT
from depression_mapping_tools.utils import Cols, load_nifti

DATA_CSV = Path(__file__).parents[1] / "a_collect_image_data.csv"
OUTPUT_FOLDER = Path(__file__).parent / "descriptive_images"

SDSM_BINARY_THRESHOLD = 0.6

# %%
# load df and drop excluded subjects
data_df = pd.read_csv(DATA_CSV)
data_df_trauma = data_df.copy()
data_df_trauma = data_df_trauma[
    data_df_trauma[Cols.EXCLUSION_REASON] == TRAUMA_EXCLUSION_COMMENT
]
data_df = data_df[data_df[Cols.EXCLUDED] == 0]

# create blanks in first iteration; get shape from example files
row = data_df.iloc[0, :]

example_lesion: Nifti1Image = load_nifti(row[Cols.PATH_LESION_IMAGE])
lesion_overlap_array = np.zeros(example_lesion.shape, dtype=np.uint16)
example_lnm: Nifti1Image = load_nifti(row[Cols.PATH_LNM_IMAGE])
lnm_mean_array = np.zeros(example_lnm.shape, dtype=np.float32)
example_disc: Nifti1Image = load_nifti(row[Cols.PATH_DISCMAP_IMAGE])
disc_overlap_array = np.zeros(example_disc.shape, dtype=np.uint16)

n_lnm_maps = data_df[Cols.PATH_LNM_IMAGE].notna().sum()

# %%
for i, (_, row) in enumerate(data_df.iterrows()):
    # create blanks in first iteration; get shape from example files
    if i == 0:
        example_lesion: Nifti1Image = load_nifti(row[Cols.PATH_LESION_IMAGE])
        lesion_overlap_array = np.zeros(example_lesion.shape, dtype=np.uint16)
        example_lnm: Nifti1Image = load_nifti(row[Cols.PATH_LNM_IMAGE])
        lnm_mean_array = np.zeros(example_lnm.shape, dtype=np.float32)
        example_disc: Nifti1Image = load_nifti(row[Cols.PATH_DISCMAP_IMAGE])
        disc_overlap_array = np.zeros(example_disc.shape, dtype=np.uint16)

    # lesion
    lesion_img: Nifti1Image = load_nifti(row[Cols.PATH_LESION_IMAGE])
    lesion_arr = lesion_img.get_fdata()
    lesion_arr = lesion_arr > 0

    lesion_overlap_array += lesion_arr.astype(np.uint16)

    # disconnection
    disc_img: Nifti1Image = load_nifti(row[Cols.PATH_DISCMAP_IMAGE])
    disc_arr = disc_img.get_fdata()  # type: ignore
    disc_arr = disc_arr > SDSM_BINARY_THRESHOLD

    disc_overlap_array += disc_arr.astype(np.uint16)

    # lesion network maps
    # only process if LNM is available for patient
    if pd.notna(row[Cols.PATH_LNM_IMAGE]):
        lnm_img: Nifti1Image = load_nifti(row[Cols.PATH_LNM_IMAGE])
        lnm_arr = lnm_img.get_fdata()
        # divide by total nubmer of patients and sum up
        weighted_lnm_arr = lnm_arr / n_lnm_maps
        lnm_mean_array += weighted_lnm_arr.astype(np.float32)

# %%
# store results, adapt headers from example images
OUTPUT_FOLDER.mkdir(exist_ok=True)

# lesion
lesion_max_val = lesion_overlap_array.max()
lesion_affine = example_lesion.affine
lesion_header = example_lesion.header.copy()
lesion_header.set_data_dtype(lesion_overlap_array.dtype)

lesion_img = Nifti1Image(
    lesion_overlap_array, affine=lesion_affine, header=lesion_header
)
filename = OUTPUT_FOLDER / f"lesion_overlap_max{lesion_max_val}.nii.gz"
nib.save(lesion_img, filename)  # type: ignore

# disconnection
disc_max_val = disc_overlap_array.max()
disc_affine = example_disc.affine
disc_header = example_disc.header.copy()
disc_header.set_data_dtype(disc_overlap_array.dtype)

disc_img = Nifti1Image(disc_overlap_array, affine=disc_affine, header=disc_header)
filename = OUTPUT_FOLDER / f"disc_overlap_max{disc_max_val}.nii.gz"
nib.save(disc_img, filename)  # type: ignore

# lesion network maps
lnm_affine = example_lnm.affine
lnm_header = example_lnm.header.copy()
lnm_header.set_data_dtype(lnm_mean_array.dtype)

lnm_img = Nifti1Image(lnm_mean_array, affine=lnm_affine, header=lnm_header)
filename = OUTPUT_FOLDER / "lnm_mean.nii.gz"
nib.save(lnm_img, filename)  # type: ignore

# %%
# create overlap of traumatic lesions for exploration
lesion_overlap_array_trauma = np.zeros(example_lesion.shape, dtype=np.uint16)

for _, row in data_df_trauma.iterrows():
    lesion_img: Nifti1Image = load_nifti(row[Cols.PATH_LESION_IMAGE])
    lesion_arr = lesion_img.get_fdata()
    lesion_arr = lesion_arr > 0

    lesion_overlap_array_trauma += lesion_arr.astype(np.uint16)

lesion_max_val_trauma = lesion_overlap_array_trauma.max()
lesion_img_trauma = Nifti1Image(
    lesion_overlap_array_trauma, affine=lesion_affine, header=lesion_header
)
filename = OUTPUT_FOLDER / f"lesion_overlap_trauma_max{lesion_max_val_trauma}.nii.gz"
nib.save(lesion_img_trauma, filename)  # type: ignore

# %%
