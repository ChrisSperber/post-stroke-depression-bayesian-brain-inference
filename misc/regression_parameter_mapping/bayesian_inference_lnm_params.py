"""Map regression beta parameters for Lesion Deficit Inference for LNMs.

Requirements:
- CSV listing all included cases and depression scores generated with a_collect_image_data.py

Outputs:
- map of raw beta parameters
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
    BETA_PARAMETER_MAP_OUTDIR,
    TRAUMA_EXCLUSION_COMMENT,
)
from depression_mapping_tools.utils import (
    Cols,
    CorrelationFormat,
    SampleSelectionMode,
    load_nifti,
    power_transform,
    run_voxelwise_beta_param_map,
)

PEARSON_R_TRANSFORM = CorrelationFormat.ARTANH_PEARSON

# choose an image that should define the format of the results file. All images with differing
# format are transformed into this image space; also, the output will have this shape
REFERENCE_LNM_SUBJECT_ID = "BBS001"

OUTPUT_DIR_BASE = "Output_LNM"

# Set to STROKE for standard sample, or STROKE_TRAUMA for stroke sample extended with traumata
SAMPLE_MODE = SampleSelectionMode.STROKE

# %%
data = pd.read_csv(Path(__file__).parents[2] / "a_collect_image_data.csv")

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
reference_nifti = load_nifti(reference_lnm_path)  # pyright: ignore[reportArgumentType]

# ensure Output directory exists
BETA_PARAMETER_MAP_OUTDIR.mkdir(parents=True, exist_ok=True)

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

beta_map = run_voxelwise_beta_param_map(
    image_data_4d=all_lnm,
    target_var=data[Cols.DEPRESSION_SCORE],  # pyright: ignore[reportArgumentType]
    minimum_analysis_threshold=None,
    n_jobs=-1,
)

# %%


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
        BETA_PARAMETER_MAP_OUTDIR / f"{OUTPUT_DIR_BASE}_{transform_string}_{timestamp}"
    )
elif SAMPLE_MODE == SampleSelectionMode.STROKE_TRAUMA:
    output_dir = (
        BETA_PARAMETER_MAP_OUTDIR
        / AETIOLOGY_SENSITIVITY_ANALYSIS_SUBDIR
        / f"{OUTPUT_DIR_BASE}_{transform_string}_{timestamp}"
    )
else:
    raise ValueError(f"Unknown Sample Mode {SAMPLE_MODE}")
output_dir.mkdir(parents=True, exist_ok=True)

beta_map_full = Nifti1Image(beta_map, affine=affine, header=header_float32)
filename = output_dir / f"BF_full_lnm_{timestamp}.nii.gz"
beta_map_full.to_filename(str(filename))

# %%
# store meta data on the analysis
image_shape = beta_map.shape
shape_str = ",".join(map(str, image_shape))

params = {
    "Analysis": "Beta parameter mapping",
    "timestamp": timestamp,
    "n_subjects": data.shape[0],
    "aetiology_selected": SAMPLE_MODE.value,
    "image_shape": shape_str,
    "pearson_transform": PEARSON_R_TRANSFORM.value,
}


with open(output_dir / f"analysis_params_lnm_{timestamp}.txt", "w") as f:
    for key, value in params.items():
        f.write(f"{key}: {value}\n")

# %%
