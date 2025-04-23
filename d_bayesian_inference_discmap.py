"""Perform Bayesian Lesion Deficit Inference for Disconnection Maps.

Voxel-wise lesion-deficit inference is performed for LNMs. Output is a map of Bayes Factors
which are approximated via the BICs of General Linear Models as described in
Wagenmakers, E. J. (2007). A practical solution to the pervasive problems of p values.
Psychonomic bulletin & review, 14(5), 779-804

Images were equally sized and no adaptions were required.

Image values are voxel-wise disconnection probabilites. The global variable DISCONNECTION_FORMAT
allows setting the data format used in the analysis (binary/continuous).

Given the 1x1x1mm³ resolution and float data format, a 16GB RAM system was not able to load all
images at once. As a workaround, the data were masked to only include relevant voxels and stored in
an patients x voxels 2D array.

Requirements:
- CSV listing all included cases generated with a_collect_image_data.py
- a_collect_image_data.py must list the paths to local maps and depression scores

Outputs:
- Bayes Factor map
- Binned Bayes Factor map (following the visualisation in 10.1016/j.neuroimage.2023.120008)
- txt with additional information
"""

# %%
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.utils import DisconnectionFormat, bin_bf_map, run_voxelwise_bf_map_2d

DISCONNECTION_FORMAT = DisconnectionFormat.CONTINUOUS  # set processing mode
BINARY_THRESHOLD = (
    0.6  # if DisconnectionFormat.BINARY, all values >= this value are set to 1 else 0
)

# Define the minimum amount of disconnection per voxel to be included in the analysis
# Only applies to analysis of binarised disconnection maps
MIN_DISCONNECTION_ANALYSIS_THRESHOLD = 10

OUTPUT_DIR_PARENT = Path(__file__).parent / "BLDI_OUTPUTS"

# choose an image that should define the format of the results file. This serves as a reference.
REFERENCE_DISCMAP_SUBJECT_ID = "BBS001"

SUBJECT_ID = "SubjectID"
DEPRESSION_SCORE = "DepressionZScore"
EXCLUDED = "Excluded"
PATH_DISCMAP_IMAGE = "PathDiscMapImage"
OUTPUT_DIR_BASE = "Output_SDSM"

# %%
data = pd.read_csv(Path(__file__).parent / "a_collect_image_data.csv")
data = data[data[EXCLUDED] == 0]

# ensure float type of scores
data[DEPRESSION_SCORE] = pd.to_numeric(data[DEPRESSION_SCORE], errors="coerce")

# get the lesion path of the reference lesion
reference_discmap_path = data.loc[
    data[SUBJECT_ID] == REFERENCE_DISCMAP_SUBJECT_ID, PATH_DISCMAP_IMAGE
].values[0]
reference_nifti: nib.Nifti1Image = nib.load(reference_discmap_path)

# ensure Output directory exists
OUTPUT_DIR_PARENT.mkdir(parents=True, exist_ok=True)

# %% create overlap map and derive analysis mask
# the full images of >2000 subjects at 182x218x182 in float format was not processable with a 16GB
# RAM system. As a workaround, the data are masked to only include voxels inside the brain and
# vectorised; the final analysis is performed on 2D (instead of 4D) data.
SDSM_ANALYSIS_MASK_PATH = Path(__file__).parent / "utils" / "sdsm_analysis_mask.nii.gz"
if SDSM_ANALYSIS_MASK_PATH.exists():
    print("An analysis mask was found and is loaded from")
    print(f"{SDSM_ANALYSIS_MASK_PATH.as_posix()}")

    analysis_mask_nifti: nib.Nifti1Image = nib.load(SDSM_ANALYSIS_MASK_PATH)
    analysis_mask_array = analysis_mask_nifti.get_fdata().astype(np.uint8)
else:
    # create analysis mask
    print("No analysis mask found; creating new mask")
    file_paths = data.loc[:, PATH_DISCMAP_IMAGE]
    overlap_array = np.zeros(reference_nifti.shape).astype(np.uint16)

    for path in tqdm(file_paths, desc="Loading DiscMaps to create analysis mask"):
        nifti: nib.Nifti1Image = nib.load(path)
        img_array = nifti.get_fdata().astype(np.float32)
        img_array_binary = (img_array != 0).astype(np.uint8)
        overlap_array = overlap_array + img_array_binary

    # retain all voxels that carry a non-zero value at least once in the dataset
    analysis_mask_array = (overlap_array != 0).astype(np.uint8)

    affine = reference_nifti.affine
    header_uint8 = reference_nifti.header.copy()
    header_uint8.set_data_dtype(np.uint8)
    analysis_mask_nifti = nib.Nifti1Image(
        analysis_mask_array, affine=affine, header=header_uint8
    )
    filename = SDSM_ANALYSIS_MASK_PATH
    analysis_mask_nifti.to_filename(str(filename))

analysis_mask_array = analysis_mask_array.astype(bool)


# %%
# load lnm images
file_paths = data.loc[:, PATH_DISCMAP_IMAGE]

n_subjects = len(file_paths)
n_voxels = np.sum(analysis_mask_array)

all_discmaps_vectorised = np.zeros((n_subjects, n_voxels), dtype=np.float32)

for i, path in enumerate(file_paths):
    img: nib.Nifti1Image = nib.load(path)
    img_data = img.get_fdata(dtype=np.float32)
    masked_data = img_data[analysis_mask_array]
    all_discmaps_vectorised[i] = masked_data

print("All DiscMap images were succesfully loaded")

# %% transform data according to DISCONNECTION_FORMAT and set minimum_threshold
if DISCONNECTION_FORMAT == DisconnectionFormat.BINARY:
    print(f"Disconnection maps are binarised at threshold >= {BINARY_THRESHOLD}")
    all_discmaps_vectorised = (all_discmaps_vectorised >= BINARY_THRESHOLD).astype(int)

    minimum_threshold = MIN_DISCONNECTION_ANALYSIS_THRESHOLD
elif DISCONNECTION_FORMAT == DisconnectionFormat.CONTINUOUS:
    minimum_threshold = None
else:
    raise ValueError("Unknown disconnection format")

# %%
# Analysis
print("Starting analysis. This may take several minutes.")

bf_map_masked_vector = run_voxelwise_bf_map_2d(
    image_data_2d=all_discmaps_vectorised,
    target_var=data[DEPRESSION_SCORE],
    minimum_analysis_threshold=minimum_threshold,
    n_jobs=-1,
)

# %%
# recreate 3D image from vector
bf_map = np.zeros_like(analysis_mask_array, dtype=np.float32)
bf_map[analysis_mask_array] = bf_map_masked_vector

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
format_string = DISCONNECTION_FORMAT.value.lower()
OUTPUT_DIR = OUTPUT_DIR_PARENT / f"{OUTPUT_DIR_BASE}_{format_string}_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

bf_map_full = nib.Nifti1Image(bf_map, affine=affine, header=header_float32)
filename = OUTPUT_DIR / f"BF_full_discmaps_{timestamp}.nii.gz"
bf_map_full.to_filename(str(filename))

bf_map_h1 = nib.Nifti1Image(
    binned_bf_maps.bf_map_h1, affine=affine, header=header_uint8
)
filename = OUTPUT_DIR / f"BF_h1_discmaps_{timestamp}.nii.gz"
bf_map_h1.to_filename(str(filename))

bf_map_h0 = nib.Nifti1Image(
    binned_bf_maps.bf_map_h0, affine=affine, header=header_uint8
)
filename = OUTPUT_DIR / f"BF_h0_discmaps_{timestamp}.nii.gz"
bf_map_h0.to_filename(str(filename))

bf_map_noev = nib.Nifti1Image(
    binned_bf_maps.bf_map_noev, affine=affine, header=header_uint8
)
filename = OUTPUT_DIR / f"BF_noev_discmaps_{timestamp}.nii.gz"
bf_map_noev.to_filename(str(filename))

# %%
# store meta data on the analysis
image_shape = bf_map.shape
shape_str = ",".join(map(str, image_shape))

voxel_count = np.count_nonzero(bf_map > 0)
if DISCONNECTION_FORMAT == DisconnectionFormat.BINARY:
    disconnection_threshold = BINARY_THRESHOLD
else:
    disconnection_threshold = "N/A"

params = {
    "Analysis": "Bayesian GLM via BIC - Bayes Factor Approximation",
    "timestamp": timestamp,
    "n_subjects": data.shape[0],
    "image_shape": shape_str,
    "analysed_voxels": voxel_count,
    "minimum_BF": np.min(bf_map[bf_map > 0]),
    "maximum_BF": np.max(bf_map),
    "disconnection_format": DISCONNECTION_FORMAT.value,
    "binarisation_threshold": disconnection_threshold,
}


with open(OUTPUT_DIR / f"analysis_params_discmaps_{timestamp}.txt", "w") as f:
    for key, value in params.items():
        f.write(f"{key}: {value}\n")

# %%
