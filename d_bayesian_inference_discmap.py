"""Perform Bayesian Lesion Deficit Inference for Disconnection Maps.

Voxel-wise lesion-deficit inference is performed for LNMs. Output is a map of Bayes Factors
which are approximated via the BICs of General Linear Models as described in
Wagenmakers, E. J. (2007). A practical solution to the pervasive problems of p values.
Psychonomic bulletin & review, 14(5), 779-804

Images were equally sized and no adaptions were required.

Image values are voxel-wise disconnection probabilites. The global variable DISCONNECTION_FORMAT
allows setting the data format (binary/continuous).

Requirements:
- CSV listing all included cases generated with a_collect_image_data.py
- a_collect_image_data.py must list the paths to local maps and depression scores

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
from tqdm import tqdm

from utils.utils import DisconnectionFormat, bin_bf_map, run_voxelwise_bf_map

DISCONNECTION_FORMAT = DisconnectionFormat.BINARY  # set for processing mode
BINARY_THRESHOLD = (
    0.6  # if DisconnectionFormat.BINARY, all values >= this value are set to 1 else 0
)

# Define the minimum amount of disconnection per voxel to be included in the analysis
# Only applies to analysis of binarised disconnection maps!
MIN_DISCONNECTION_ANALYSIS_THRESHOLD = 10

OUTPUT_DIR = Path(__file__).parent / "BLDI_OUTPUTS"

# choose an image that should define the format of the results file. This serves as a reference.
REFERENCE_DISCMAP_SUBJECT_ID = "BBS001"

SUBJECT_ID = "SubjectID"
DEPRESSION_SCORE = "DepressionZScore"
EXCLUDED = "Excluded"
PATH_DISCMAP_IMAGE = "PathDiscMapImage"

# %%
data = pd.read_csv(Path(__file__).parent / "a_collect_image_data.csv")
data = data[data[EXCLUDED] == 0]

# get the lesion path of the reference lesion
reference_discmap_path = data.loc[
    data[SUBJECT_ID] == REFERENCE_DISCMAP_SUBJECT_ID, PATH_DISCMAP_IMAGE
].values[0]
reference_nifti = nib.load(reference_discmap_path)

# ensure Output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# load lnm images
file_paths = data.loc[:, PATH_DISCMAP_IMAGE]
all_discmaps_list = []

for path in tqdm(file_paths, desc="Loading DiscMap NifTi"):
    nifti = nib.load(path)
    img_array = nifti.get_fdata().astype(np.float32)
    all_discmaps_list.append(img_array)

# Stack into 4D array: (N_images, X, Y, Z)
all_discmaps = np.stack(all_discmaps_list, axis=0)
# cleanup
del all_discmaps_list
gc.collect()

print("All DiscMap images were succesfully loaded")

# %% transform data according to DISCONNECTION_FORMAT

if DISCONNECTION_FORMAT == DisconnectionFormat.BINARY:
    print(f"Disconnection maps are binarised at threshold >= {BINARY_THRESHOLD}")
    all_discmaps = (all_discmaps >= BINARY_THRESHOLD).astype(int)

# %%
# Analysis
print("Starting analysis. This may take several minutes.")

# set minimum threshold if analysis in binary
if DISCONNECTION_FORMAT == DisconnectionFormat.BINARY:
    minimum_threshold = MIN_DISCONNECTION_ANALYSIS_THRESHOLD
elif DISCONNECTION_FORMAT == DisconnectionFormat.CONTINUOUS:
    minimum_threshold = None
else:
    raise ValueError("Unknown disconnection format")

bf_map = run_voxelwise_bf_map(
    image_data_4d=all_discmaps,
    target_var=data[DEPRESSION_SCORE],
    minimum_analysis_threshold=minimum_threshold,
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

params = {
    "Analysis": "Bayesian GLM via BIC - Bayes Factor Approximation",
    "timestamp": timestamp,
    "n_subjects": data.shape[0],
    "image_shape": shape_str,
    "analysed_voxels": voxel_count,
    "minimum_BF": np.min(bf_map[bf_map > 0]),
    "maximum_BF": np.max(bf_map),
    "disconnection_format": DISCONNECTION_FORMAT.value,
}


with open(OUTPUT_DIR / f"analysis_params_discmaps_{timestamp}.txt", "w") as f:
    for key, value in params.items():
        f.write(f"{key}: {value}\n")

# %%
