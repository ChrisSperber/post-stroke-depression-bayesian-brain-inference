"""Embed the resulting maps into brain atlases for interpretation.

Requirements:
    - Bayesian statistical result maps were generated with the previous scripts b to d
    - atlasses were downloaded with script e

Outputs:
    - json with detailed results, showing the contents of multiple results dfs
    - smaller tables with condensed results are printed to the terminal for copypasting

"""

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
from nibabel.nifti1 import Nifti1Image
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img

from depression_mapping_tools.config import BLDI_OUTPUT_DIR_PARENT
from depression_mapping_tools.utils import assign_segmentation2regions, load_nifti

BRAIN_ATLAS_DIR = Path(__file__).parent / "brain_atlasses"

# Define final results files for Lesion, LNM, and SDSM
FINAL_RESULT_LESION_FOLDER = "Output_Lesion_20260112_0032"
FINAL_RESULT_LNM_FOLDER = "Output_LNM_artanh_pearson_r_20260112_1119"
FINAL_RESULT_SDSM_FOLDER = "Output_SDSM_binarised_disconnection_20260112_0912"

REGION_LABEL = "region_label"
VOLUME_MM3 = "volume_mm3"
VOXEL_COUNT = "voxel_count"
BACKGROUND = "Background"  # Background label in Harvard-Oxford atlas
FIBRE_NAME = "fibre_name"
FIBRE_PATH = "fibre_file_path"

# interpretation of results is done with binarised statistical topographies focussing on evidence
# for H1, i.e. large Bayes Factors >1. Due to differently large peaks, statistical results are
# treated differently, with a low threshold for lesion and lnm, and a high threshold for sdsm, as
# the results there had extensive and very strong evidence for H1
THRESHOLD_BLDI_BINARISATION_LOW = 3
THRESHOLD_BLDI_BINARISATION_HIGH = 100

# fibre maps from the Atlas of Human Brain Connections are thresholded to compute overlaps with
# the statistical results
THRESHOLD_FIBRE_ATLAS = 0.5

# %%
harvard_oxford_atlas_cort = fetch_atlas_harvard_oxford(
    atlas_name="cortl-maxprob-thr25-1mm", symmetric_split=True
)
harvard_oxford_atlas_subcort = fetch_atlas_harvard_oxford(
    atlas_name="sub-maxprob-thr25-1mm", symmetric_split=True
)

fibres_dir = BRAIN_ATLAS_DIR / "BCB_Fibres" / "fibres" / "Tracts"

# %%
# load results maps as NIFTIs
results_folder = BLDI_OUTPUT_DIR_PARENT / FINAL_RESULT_LESION_FOLDER
(filename,) = [f for f in results_folder.iterdir() if f.is_file() and "full" in f.name]
result_nifti_lesion: Nifti1Image = load_nifti(filename)
result_arr_lesion = result_nifti_lesion.get_fdata()

results_folder = BLDI_OUTPUT_DIR_PARENT / FINAL_RESULT_LNM_FOLDER
(filename,) = [f for f in results_folder.iterdir() if f.is_file() and "full" in f.name]
result_nifti_lnm: Nifti1Image = load_nifti(filename)
result_arr_lnm = result_nifti_lnm.get_fdata()

results_folder = BLDI_OUTPUT_DIR_PARENT / FINAL_RESULT_SDSM_FOLDER
(filename,) = [f for f in results_folder.iterdir() if f.is_file() and "full" in f.name]
result_nifti_sdsm: Nifti1Image = load_nifti(filename)
result_arr_sdsm = result_nifti_sdsm.get_fdata()

# %%
# interpretation of lesion results via Harvard-Oxford Atlas
ho_atlas_nifti_cort: Nifti1Image = harvard_oxford_atlas_cort.maps
ho_atlas_nifti_subcort: Nifti1Image = harvard_oxford_atlas_subcort.maps

# HO Atlas is in RAS orientation, the analysed data in LAS; hence re-orientation is required
ho_atlas_nifti_cort_las = resample_to_img(
    ho_atlas_nifti_cort,
    result_nifti_lesion,
    interpolation="nearest",
    force_resample=True,
    copy_header=True,
)
ho_atlas_nifti_subcort_las = resample_to_img(
    ho_atlas_nifti_subcort,
    result_nifti_lesion,
    interpolation="nearest",
    force_resample=True,
    copy_header=True,
)

ho_atlas_arr_cort = ho_atlas_nifti_cort_las.get_fdata()
ho_atlas_arr_subcort = ho_atlas_nifti_subcort_las.get_fdata()

# compute region-wise overlaps and create results dfs
bldi_results_lesion_binary_arr = (
    result_arr_lesion >= THRESHOLD_BLDI_BINARISATION_LOW
).astype(np.uint8)
atlas_comparison_lesion_cort = assign_segmentation2regions(
    binary_segmentation_arr=bldi_results_lesion_binary_arr, atlas_arr=ho_atlas_arr_cort
)
atlas_comparison_lesion_cort[REGION_LABEL] = harvard_oxford_atlas_cort.labels

atlas_comparison_lesion_subcort = assign_segmentation2regions(
    binary_segmentation_arr=bldi_results_lesion_binary_arr,
    atlas_arr=ho_atlas_arr_subcort,
)
atlas_comparison_lesion_subcort[REGION_LABEL] = harvard_oxford_atlas_subcort.labels

# print condensed results
# add mm³ column
voxel_dims = result_nifti_lesion.header.get_zooms()[:3]
conversion_factor = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]  # type: ignore
atlas_comparison_lesion_cort[VOLUME_MM3] = (
    (atlas_comparison_lesion_cort[VOXEL_COUNT] * conversion_factor).round().astype(int)
)
atlas_comparison_lesion_subcort[VOLUME_MM3] = (
    (atlas_comparison_lesion_subcort[VOXEL_COUNT] * conversion_factor)
    .round()
    .astype(int)
)

atlas_comparison_lesion_cort_top10 = (
    atlas_comparison_lesion_cort[
        atlas_comparison_lesion_cort[REGION_LABEL] != BACKGROUND
    ]
    .sort_values(VOLUME_MM3, ascending=False)
    .head(10)[[REGION_LABEL, VOLUME_MM3]]
)
print(atlas_comparison_lesion_cort_top10.to_csv(sep="\t", index=False))

atlas_comparison_lesion_subcort_top10 = (
    atlas_comparison_lesion_subcort[
        atlas_comparison_lesion_subcort[REGION_LABEL] != BACKGROUND
    ]
    .sort_values(VOLUME_MM3, ascending=False)
    .head(10)[[REGION_LABEL, VOLUME_MM3]]
)
print(atlas_comparison_lesion_subcort_top10.to_csv(sep="\t", index=False))

# %%
# interpretation of lesion-network results via Harvard-Oxford Atlas
# HO Atlas is in 1x1x1 resolution, results in 2x2x2, requiring interpolation
ho_atlas_nifti_cort_las2mm = resample_to_img(
    ho_atlas_nifti_cort,
    result_nifti_lnm,
    interpolation="nearest",
    force_resample=True,
    copy_header=True,
)
ho_atlas_nifti_subcort_las2mm = resample_to_img(
    ho_atlas_nifti_subcort,
    result_nifti_lnm,
    interpolation="nearest",
    force_resample=True,
    copy_header=True,
)
ho_atlas_arr_cort_las2mm = ho_atlas_nifti_cort_las2mm.get_fdata()
ho_atlas_arr_subcort_las2mm = ho_atlas_nifti_subcort_las2mm.get_fdata()

# compute region-wise overlaps and create results dfs
bldi_results_lnm_binary_arr = (
    result_arr_lnm >= THRESHOLD_BLDI_BINARISATION_LOW
).astype(np.uint8)
atlas_comparison_lnm_cort = assign_segmentation2regions(
    binary_segmentation_arr=bldi_results_lnm_binary_arr,
    atlas_arr=ho_atlas_arr_cort_las2mm,
)
atlas_comparison_lnm_cort[REGION_LABEL] = harvard_oxford_atlas_cort.labels

atlas_comparison_lnm_subcort = assign_segmentation2regions(
    binary_segmentation_arr=bldi_results_lnm_binary_arr,
    atlas_arr=ho_atlas_arr_subcort_las2mm,
)
atlas_comparison_lnm_subcort[REGION_LABEL] = harvard_oxford_atlas_subcort.labels

# print condensed results
# add mm³ column
voxel_dims = result_nifti_lnm.header.get_zooms()[:3]
conversion_factor = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]  # type: ignore
atlas_comparison_lnm_cort[VOLUME_MM3] = (
    (atlas_comparison_lnm_cort[VOXEL_COUNT] * conversion_factor).round().astype(int)
)
atlas_comparison_lnm_subcort[VOLUME_MM3] = (
    (atlas_comparison_lnm_subcort[VOXEL_COUNT] * conversion_factor).round().astype(int)
)

atlas_comparison_lnm_cort_top10 = (
    atlas_comparison_lnm_cort[atlas_comparison_lnm_cort[REGION_LABEL] != BACKGROUND]
    .sort_values(VOLUME_MM3, ascending=False)
    .head(10)[[REGION_LABEL, VOLUME_MM3]]
)
print(atlas_comparison_lnm_cort_top10.to_csv(sep="\t", index=False))

atlas_comparison_lnm_subcort_top10 = (
    atlas_comparison_lnm_subcort[
        atlas_comparison_lnm_subcort[REGION_LABEL] != BACKGROUND
    ]
    .sort_values(VOLUME_MM3, ascending=False)
    .head(10)[[REGION_LABEL, VOLUME_MM3]]
)
print(atlas_comparison_lnm_subcort_top10.to_csv(sep="\t", index=False))

# %%
# interpretation of disconnectome maps via BCB tract maps
bldi_results_sdsm_binary_arr = (
    result_arr_sdsm >= THRESHOLD_BLDI_BINARISATION_HIGH
).astype(np.uint8)

fibre_files = [
    f for f in fibres_dir.iterdir() if f.is_file() and f.name.endswith(".nii.gz")
]
fibre_comparison_sdsm = pd.DataFrame(
    {
        FIBRE_PATH: [f.resolve() for f in fibre_files],
        FIBRE_NAME: [f.stem for f in fibre_files],
    }
)

voxel_dims = result_nifti_sdsm.header.get_zooms()[:3]
conversion_factor = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]  # type: ignore

# loop through fibre maps
for index, row in fibre_comparison_sdsm.iterrows():
    # load fibre and reorient
    fibre_nifti: Nifti1Image = load_nifti(row[FIBRE_PATH])
    fibre_nifti = resample_to_img(
        fibre_nifti,
        result_nifti_sdsm,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    # get array data and binarise
    fibre_nifti_arr = fibre_nifti.get_fdata()
    fibre_nifti_arr_binary = (fibre_nifti_arr >= THRESHOLD_FIBRE_ATLAS).astype(np.uint8)

    fibre_voxel_count = (
        np.logical_and(fibre_nifti_arr_binary, bldi_results_sdsm_binary_arr)
        .sum()
        .astype(int)
    )
    fibre_comparison_sdsm.loc[index, VOXEL_COUNT] = fibre_voxel_count  # type: ignore
    fibre_volume_mm3 = int(round(fibre_voxel_count * conversion_factor))
    fibre_comparison_sdsm.loc[index, VOLUME_MM3] = fibre_volume_mm3  # type: ignore

fibre_comparison_sdsm[VOLUME_MM3] = fibre_comparison_sdsm[VOLUME_MM3].astype(int)
fibre_comparison_sdsm[FIBRE_NAME] = fibre_comparison_sdsm[FIBRE_NAME].str.removesuffix(
    ".nii"
)

# print condensed results
fibre_comparison_sdsm_top15 = fibre_comparison_sdsm.sort_values(
    VOLUME_MM3, ascending=False
).head(15)[[FIBRE_NAME, VOLUME_MM3]]
print(fibre_comparison_sdsm_top15.to_csv(sep="\t", index=False))

# remove paths from df to make it serialisable
fibre_comparison_sdsm = fibre_comparison_sdsm.drop(FIBRE_PATH, axis=1)

# %%
# store detailed results
results = {
    "Results lesion mapping cortical HO atlas": atlas_comparison_lesion_cort.to_dict(
        orient="records"
    ),
    "Results lesion mapping subcortical HO atlas": atlas_comparison_lesion_subcort.to_dict(
        orient="records"
    ),
    "Results lnm mapping cortical HO atlas": atlas_comparison_lnm_cort.to_dict(
        orient="records"
    ),
    "Results lnm mapping subcortical HO atlas": atlas_comparison_lnm_subcort.to_dict(
        orient="records"
    ),
    "Results sdsm mapping fibre anatomy atlas": fibre_comparison_sdsm.to_dict(
        orient="records"
    ),
}
output_name = Path(__file__).with_suffix(".json")
with open(output_name, "w") as f:
    json.dump(results, f, indent=2)

# %%
