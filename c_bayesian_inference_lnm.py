"""Perform Bayesian Lesion Deficit Inference for LNMs.

Voxel-wise lesion-deficit inference is performed for LNMs. Output is a map of Bayes Factors
which are approximated via the BICs of General Linear Models as described in
Wagenmakers, E. J. (2007). A practical solution to the pervasive problems of p values.
Psychonomic bulletin & review, 14(5), 779-804

Images were equally sized and no adaptions were required.

Image values are Pearson correlation coefficients that were artanh-transformed (known as Fisher
transformation). This is not necessarily the best way to represent the network, as large correlation
scores - which are only found in the lesion areas where autocorrelation is present - are further
increased and thereby inflated. The global variable CORRELATION_FORMAT allows setting alternatives.

For some patients, no LNMs were available as lesions were restricted to white matter. These were not
exlcuded from the main sample and have to be excluded here.

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

import numpy as np
import pandas as pd
from nibabel.nifti1 import Nifti1Image
from tqdm import tqdm

from depression_mapping_tools.config import (
    AETIOLOGY_SENSITIVITY_ANALYSIS_SUBDIR,
    BLDI_OUTPUT_DIR_PARENT,
    TRAUMA_EXCLUSION_COMMENT,
)
from depression_mapping_tools.utils import (
    Cols,
    CorrelationFormat,
    SampleSelectionMode,
    bin_bf_map,
    load_nifti,
    power_transform,
    run_voxelwise_bf_map,
)

PEARSON_R_TRANSFORM = CorrelationFormat.ARTANH_PEARSON

# choose an image that should define the format of the results file. All images with differing
# format are transformed into this image space; also, the output will have this shape
REFERENCE_LNM_SUBJECT_ID = "BBS001"

OUTPUT_DIR_BASE = "Output_LNM"

# Set to STROKE for standard sample, or STROKE_TRAUMA for stroke sample extended with traumata
SAMPLE_MODE = SampleSelectionMode.STROKE

# %%
data = pd.read_csv(Path(__file__).parent / "a_collect_image_data.csv")

if SAMPLE_MODE == SampleSelectionMode.STROKE:
    data = data[data[Cols.EXCLUDED] == 0]
elif SAMPLE_MODE == SampleSelectionMode.STROKE_TRAUMA:
    data = data[
        (data[Cols.EXCLUDED] == 0)
        | (data[Cols.EXCLUSION_REASON] == TRAUMA_EXCLUSION_COMMENT)
    ]
else:
    msg = f"Unknown Sample selection mode {SAMPLE_MODE}"
    raise ValueError(msg)

# additionally exclude cases with missing LNMs
data = data[data[Cols.PATH_LNM_IMAGE].notna()]

# ensure float type of scores
data[Cols.DEPRESSION_SCORE] = pd.to_numeric(
    data[Cols.DEPRESSION_SCORE], errors="coerce"
)

# get the lesion path of the reference lesion
reference_lnm_path = data.loc[
    data[Cols.SUBJECT_ID] == REFERENCE_LNM_SUBJECT_ID, Cols.PATH_LNM_IMAGE
].values[0]
reference_nifti = load_nifti(reference_lnm_path)

# ensure Output directory exists
BLDI_OUTPUT_DIR_PARENT.mkdir(parents=True, exist_ok=True)

# %%
# load lnm images
file_paths = data.loc[:, Cols.PATH_LNM_IMAGE]
all_lnm_list = []

for path in tqdm(file_paths, desc="Loading LNM NifTi"):
    nifti = load_nifti(path)
    img_array = nifti.get_fdata().astype(np.float32)
    all_lnm_list.append(img_array)

# Stack into 4D array: (N_images, X, Y, Z)
all_lnm = np.stack(all_lnm_list, axis=0)
# cleanup
del all_lnm_list
gc.collect()

print("All LNM images were succesfully loaded")

# %% transform data according to PEARSON_R_TRANSFORM

if PEARSON_R_TRANSFORM == CorrelationFormat.ARTANH_PEARSON:
    print("Artanh transform, original input data are not changed")
elif PEARSON_R_TRANSFORM == CorrelationFormat.NONTRANSFORMED_PEARSON:
    print("Data are re-transformed into original r values")
    all_lnm = np.tanh(all_lnm)
elif PEARSON_R_TRANSFORM == CorrelationFormat.ATANH_PEARSON:
    print(
        "Data are re-transformed into original r values and then transformed via tanh"
    )
    all_lnm = np.tanh(np.tanh(all_lnm))
elif PEARSON_R_TRANSFORM == CorrelationFormat.POWER_TRANSFORM:
    print(
        "Data are re-transformed into original r values and then transformed via power transform"
    )
    all_lnm = np.tanh(all_lnm)
    all_lnm = power_transform(all_lnm)
else:
    msg = (
        "Invalied Pearson r transform for LNM values chosen. Check PEARSON_R_TRANSFORM"
    )
    raise ValueError(msg)

# %%
# Analysis
print("Starting analysis. This may take several minutes.")

bf_map = run_voxelwise_bf_map(
    image_data_4d=all_lnm,
    target_var=data[Cols.DEPRESSION_SCORE],  # pyright: ignore[reportArgumentType]
    minimum_analysis_threshold=None,
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
transform_string = PEARSON_R_TRANSFORM.value.lower()
if SAMPLE_MODE == SampleSelectionMode.STROKE:
    output_dir = (
        BLDI_OUTPUT_DIR_PARENT / f"{OUTPUT_DIR_BASE}_{transform_string}_{timestamp}"
    )
elif SAMPLE_MODE == SampleSelectionMode.STROKE_TRAUMA:
    output_dir = (
        BLDI_OUTPUT_DIR_PARENT
        / AETIOLOGY_SENSITIVITY_ANALYSIS_SUBDIR
        / f"{OUTPUT_DIR_BASE}_{transform_string}_{timestamp}"
    )
else:
    raise ValueError(f"Unknown Sample Mode {SAMPLE_MODE}")
output_dir.mkdir(parents=True, exist_ok=True)

bf_map_full = Nifti1Image(bf_map, affine=affine, header=header_float32)
filename = output_dir / f"BF_full_lnm_{timestamp}.nii.gz"
bf_map_full.to_filename(str(filename))

bf_map_h1 = Nifti1Image(binned_bf_maps.bf_map_h1, affine=affine, header=header_uint8)
filename = output_dir / f"BF_h1_lnm_{timestamp}.nii.gz"
bf_map_h1.to_filename(str(filename))

bf_map_h0 = Nifti1Image(binned_bf_maps.bf_map_h0, affine=affine, header=header_uint8)
filename = output_dir / f"BF_h0_lnm_{timestamp}.nii.gz"
bf_map_h0.to_filename(str(filename))

bf_map_noev = Nifti1Image(
    binned_bf_maps.bf_map_noev, affine=affine, header=header_uint8
)
filename = output_dir / f"BF_noev_lnm_{timestamp}.nii.gz"
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
    "aetiology_selected": SAMPLE_MODE.value,
    "image_shape": shape_str,
    "analysed_voxels": voxel_count,
    "minimum_BF": np.min(bf_map[bf_map > 0]),
    "maximum_BF": np.max(bf_map),
    "pearson_transform": PEARSON_R_TRANSFORM.value,
}


with open(output_dir / f"analysis_params_lnm_{timestamp}.txt", "w") as f:
    for key, value in params.items():
        f.write(f"{key}: {value}\n")

# %%
