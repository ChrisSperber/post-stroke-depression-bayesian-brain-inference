"""Perform Bayesian Lesion Deficit Inference for Lesions.

Voxel-wise lesion-deficit inference is performed for lesion maps. Output is a map of Bayes Factors
which are approximated via the BICs of General Linear Models as described in
Wagenmakers, E. J. (2007). A practical solution to the pervasive problems of p values.
Psychonomic bulletin & review, 14(5), 779-804

The code contains legacy functionality to handle lesion images that are differently sized or
formatted (resolution, dimensions, origin). With the data uploaded on 18/04/2025, this should not
be necessary anymore; however, the functionality is kept as it has no negative effects.
See REFERENCE_LESION_SUBJECT_ID below.

Requirements:
- CSV listing all included cases and depression scores generated with a_collect_image_data.py

Outputs:
- Bayes Factor map
- Binned Bayes Factor map (following the visualisation in 10.1016/j.neuroimage.2023.120008)
- txt with additional information
"""

# %%
import gc
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from tqdm import tqdm

from utils.utils import bin_bf_map, compare_image_affine_and_shape, run_voxelwise_bf_map

OUTPUT_DIR_PARENT = Path(__file__).parent / "BLDI_OUTPUTS"

# choose an image that should define the format of the results file. All images with differing
# format are transformed into this image space; also, the output will have this shape
REFERENCE_LESION_SUBJECT_ID = "BBS001"

# Define the minimum amount of lesions per voxel to be included in the analysis
MIN_LESION_ANALYSIS_THRESHOLD = 10

SUBJECT_ID = "SubjectID"
DEPRESSION_SCORE = "DepressionZScore"
EXCLUDED = "Excluded"
PATH_LESION_IMAGE = "PathLesionImage"
OUTPUT_DIR_BASE = "Output_Lesion"

# %%
data = pd.read_csv(Path(__file__).parent / "a_collect_image_data.csv")
data = data[data[EXCLUDED] == 0]

# ensure float type of scores
data[DEPRESSION_SCORE] = pd.to_numeric(data[DEPRESSION_SCORE], errors="coerce")

# get the lesion path of the reference lesion
reference_lesion_path = data.loc[
    data[SUBJECT_ID] == REFERENCE_LESION_SUBJECT_ID, PATH_LESION_IMAGE
].values[0]
reference_nifti = nib.load(reference_lesion_path)

# ensure Output directory exists
OUTPUT_DIR_PARENT.mkdir(parents=True, exist_ok=True)

# %%
# load lesion images in unified shape
file_paths = data.loc[:, PATH_LESION_IMAGE]
all_lesions_list = []

for path in tqdm(file_paths, desc="Loading lesion NifTi"):
    nifti = nib.load(path)

    # adapt image shape & affine if required
    if not compare_image_affine_and_shape(reference_nifti, nifti):
        nifti = resample_to_img(
            nifti, reference_nifti, interpolation="nearest", force_resample=True
        )

    img_array = nifti.get_fdata().astype(np.uint8)

    # verify that image values are binary
    is_binary = np.array_equal(img_array, img_array.astype(bool))
    if not is_binary:
        raise ValueError(f"Non-binary image {path} included.")

    all_lesions_list.append(img_array)

# Stack into 4D array: (N_images, X, Y, Z)
all_lesions = np.stack(all_lesions_list, axis=0)
# cleanup
del all_lesions_list
gc.collect()

print("All lesion images were succesfully loaded")

# %%
# Analysis
print("Starting analysis. This may take several minutes.")

bf_map = run_voxelwise_bf_map(
    image_data_4d=all_lesions,
    target_var=data[DEPRESSION_SCORE],
    minimum_analysis_threshold=MIN_LESION_ANALYSIS_THRESHOLD,
    n_jobs=-1,
)

# %%
# create binned maps for better visualisation
binned_bf_maps = bin_bf_map(bf_map)

# %%
# export results as NifTi
# the header is taken from the reference image loaded above
affine = reference_nifti.affine
header_uint8 = reference_nifti.header.copy()
header_uint8.set_data_dtype(np.uint8)
header_float32 = reference_nifti.header.copy()
header_float32.set_data_dtype(np.float32)

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_DIR = OUTPUT_DIR_PARENT / f"{OUTPUT_DIR_BASE}_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

bf_map_full = nib.Nifti1Image(bf_map, affine=affine, header=header_float32)
filename = OUTPUT_DIR / f"BF_full_lesion_{timestamp}.nii.gz"
bf_map_full.to_filename(str(filename))

bf_map_h1 = nib.Nifti1Image(
    binned_bf_maps.bf_map_h1, affine=affine, header=header_uint8
)
filename = OUTPUT_DIR / f"BF_h1_lesion_{timestamp}.nii.gz"
bf_map_h1.to_filename(str(filename))

bf_map_h0 = nib.Nifti1Image(
    binned_bf_maps.bf_map_h0, affine=affine, header=header_uint8
)
filename = OUTPUT_DIR / f"BF_h0_lesion_{timestamp}.nii.gz"
bf_map_h0.to_filename(str(filename))

bf_map_noev = nib.Nifti1Image(
    binned_bf_maps.bf_map_noev, affine=affine, header=header_uint8
)
filename = OUTPUT_DIR / f"BF_noev_lesion_{timestamp}.nii.gz"
bf_map_noev.to_filename(str(filename))

# %%
# store meta data on the analysis
image_shape = bf_map.shape
shape_str = ",".join(map(str, image_shape))

voxel_count = np.count_nonzero(bf_map > 0)

params = {
    "Analysis": "Bayesian GLM via BIC - Bayes Factor Approximation",
    "timestamp": timestamp,
    "n_subjects": data.shape[0],
    "image_shape": shape_str,
    "analysed_voxels": voxel_count,
    "minimum_analysis_threshold": MIN_LESION_ANALYSIS_THRESHOLD,
    "minimum_BF": np.min(bf_map[bf_map > 0]),
    "maximum_BF": np.max(bf_map),
}

with open(OUTPUT_DIR / f"analysis_params_lesion_{timestamp}.txt", "w") as f:
    for key, value in params.items():
        f.write(f"{key}: {value}\n")

# %%
