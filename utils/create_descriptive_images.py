"""Create Nifti Images of lesion overlap etc. for manuscript figures.

Requirements:
    - a_collect_image_data.py was run

Outputs:
    - lesion overlaps image, disconnection overlap (at binary threshold), and mean lnm map are
        generated at utils/descriptive_images

"""

# %%
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel import Nifti1Image

DATA_CSV = Path(__file__).parents[1] / "a_collect_image_data.csv"
OUTPUT_FOLDER = Path(__file__).parent / "descriptive_images"

SDSM_BINARY_THRESHOLD = 0.6

EXCLUDED = "Excluded"

PATH_LESION_IMAGE = "PathLesionImage"
PATH_LNM_IMAGE = "PathLNMImage"
PATH_DISC_IMAGE = "PathDiscMapImage"

# %%
# load df and drop excluded subjects
data_df = pd.read_csv(DATA_CSV)
data_df = data_df[data_df[EXCLUDED] == 0]

# %%
for i, (_, row) in enumerate(data_df.iterrows()):
    # create blanks in first iteration; get shape from example files
    if i == 0:
        example_lesion: Nifti1Image = nib.load(row[PATH_LESION_IMAGE])
        lesion_overlap_array = np.zeros(example_lesion.shape, dtype=np.uint16)
        example_lnm: Nifti1Image = nib.load(row[PATH_LNM_IMAGE])
        lnm_mean_array = np.zeros(example_lnm.shape, dtype=np.float32)
        example_disc: Nifti1Image = nib.load(row[PATH_DISC_IMAGE])
        disc_overlap_array = np.zeros(example_disc.shape, dtype=np.uint16)

    # lesion
    lesion_img: Nifti1Image = nib.load(row[PATH_LESION_IMAGE])
    lesion_arr = lesion_img.get_fdata()

    lesion_overlap_array += lesion_arr.astype(np.uint16)

    # disconnection
    disc_img: Nifti1Image = nib.load(row[PATH_DISC_IMAGE])
    disc_arr = disc_img.get_fdata()
    disc_arr = disc_arr > SDSM_BINARY_THRESHOLD

    disc_overlap_array += disc_arr.astype(np.uint16)

    # lesion network maps
    lnm_img: Nifti1Image = nib.load(row[PATH_LNM_IMAGE])
    lnm_arr = lnm_img.get_fdata()
    # divide by total nubmer of patients and sum up
    weighted_lnm_arr = lnm_arr / len(data_df)
    lnm_mean_array += weighted_lnm_arr.astype(np.float32)

# %%
# store results, adapt headers from example images
OUTPUT_FOLDER.mkdir(exist_ok=True)

# lesion
lesion_max_val = lesion_overlap_array.max()
lesion_affine = example_lesion.affine
lesion_header = example_lesion.header.copy()
lesion_header.set_data_dtype(lesion_overlap_array.dtype)

lesion_img = nib.Nifti1Image(
    lesion_overlap_array, affine=lesion_affine, header=lesion_header
)
filename = OUTPUT_FOLDER / f"lesion_overlap_max{lesion_max_val}.nii.gz"
nib.save(lesion_img, filename)

# disconnection
disc_max_val = disc_overlap_array.max()
disc_affine = example_disc.affine
disc_header = example_disc.header.copy()
disc_header.set_data_dtype(disc_overlap_array.dtype)

disc_img = nib.Nifti1Image(disc_overlap_array, affine=disc_affine, header=disc_header)
filename = OUTPUT_FOLDER / f"disc_overlap_max{disc_max_val}.nii.gz"
nib.save(disc_img, filename)

# lesion network maps
lnm_affine = example_lnm.affine
lnm_header = example_lnm.header.copy()
lnm_header.set_data_dtype(lnm_mean_array.dtype)

lnm_img = nib.Nifti1Image(lnm_mean_array, affine=lnm_affine, header=lnm_header)
filename = OUTPUT_FOLDER / "lnm_mean.nii.gz"
nib.save(lnm_img, filename)

# %%
