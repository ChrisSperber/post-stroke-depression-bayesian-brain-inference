"""Evaluate lesion overlap.

This script is not part of the main analysis pipeline.

Requirements:
    - create_descriptive_images.py was run
    - a Freesurfer-based segmentation of the spm mni152 T1 template is copied into the folder as
        "spm152_synthseg.nii.gz (without cortex subparcellation, i.e. without -parc flag)
    - Freesurfer label map (derived from Freesurfer, manually modified, and committed as
        freesurfer_labelmap.csv)

Outputs:
    - prints to terminal: percent of brain covered by lesions (n>0) and included in analysis
        (n>=MIN_LESION_ANALYSIS_THRESHOLD)

"""

# %%

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.nifti1 import Nifti1Image
from nilearn.image import resample_to_img

from depression_mapping_tools.config import MIN_LESION_ANALYSIS_THRESHOLD
from depression_mapping_tools.utils import load_nifti

DESCRIPTIVE_IMAGES_FOLDER = Path(__file__).parent / "descriptive_images"

MNI152_SEGM_PATH = Path(__file__).parent / "spm152_synthseg.nii.gz"

FREESURFER_LABELMAP_CSV = Path(__file__).parent / "freesurfer_labelmap.csv"

BRAIN_LABEL_STRUCTURES = ["White_Matter", "Cortex", "Subcortical_Deep_Gray"]
STRUCTURE = "Structure"
ID = "id"

CEREBELLAR_WM_LABELS = [7, 46]

# %%
# list relevant labels for brain (excluding csf, cerebellum, brainstem)
labelmap_df = pd.read_csv(FREESURFER_LABELMAP_CSV, sep=";")
labelmap_df = labelmap_df[labelmap_df[STRUCTURE].isin(BRAIN_LABEL_STRUCTURES)]
brain_labels_list = labelmap_df[ID].to_list()

brain_labels_list[:] = [x for x in brain_labels_list if x not in CEREBELLAR_WM_LABELS]

# %%
# find & load lesion overlay
matches = [
    p
    for p in DESCRIPTIVE_IMAGES_FOLDER.rglob("*.nii.gz")
    if "lesion_overlap_max" in p.name
]

if len(matches) > 1:
    raise ValueError("More than one lesion overlay found!")

overlay_nifti_path = matches[0]
overlay_nifti = load_nifti(overlay_nifti_path)

# %%
# load segmentation and resample to overlay space
segm_nifti = load_nifti(MNI152_SEGM_PATH)
segm_nifti_res: Nifti1Image = resample_to_img(
    source_img=segm_nifti, target_img=overlay_nifti, interpolation="nearest"
)

# %%
overlay_arr = overlay_nifti.get_fdata()
overlay_mask_gt_0 = (overlay_arr > 0).astype(int)
overlay_mask_gt_thres = (overlay_arr > MIN_LESION_ANALYSIS_THRESHOLD).astype(int)

# %%
# create a brain mask of the mni152
segm_arr = segm_nifti_res.get_fdata().astype(np.int32)
brain_mask = np.isin(segm_arr, np.asarray(brain_labels_list, dtype=np.int32))

# create qc image
qc_mask_path = DESCRIPTIVE_IMAGES_FOLDER / "qc_brainmask_supratentorial.nii.gz"
qc_img = Nifti1Image(
    brain_mask.astype(np.uint8), overlay_nifti.affine, overlay_nifti.header
)
qc_img.set_data_dtype(np.uint8)
nib.save(qc_img, str(qc_mask_path))  # pyright: ignore[reportPrivateImportUsage]

# %%
# print voxel overlays
brain_vox = int(brain_mask.sum())
if brain_vox == 0:
    raise ValueError("Brain mask is empty after resampling/label selection.")


def pct_covered(sub_mask: np.ndarray, brain_reference_mask: np.ndarray) -> float:
    """Percent of 'within' voxels covered by mask."""
    covered = np.logical_and(sub_mask, brain_reference_mask).sum()
    return 100.0 * float(covered) / float(brain_reference_mask.sum())


pct0 = pct_covered(overlay_mask_gt_0, brain_mask)
pctthres = pct_covered(overlay_mask_gt_thres, brain_mask)

print("\n--- Lesion overlap coverage within brain mask ---")
print(f"Brain voxels: {brain_vox:,d}")
print(f"Coverage (lesion_overlap > 0): {pct0:.1f}%")
print(f"Coverage (lesion_overlap > {MIN_LESION_ANALYSIS_THRESHOLD}): {pctthres:.1f}%")

# print absolute voxel counts for sanity check
cov0_vox = int(np.logical_and(overlay_mask_gt_0, brain_mask).sum())
covthres_vox = int(np.logical_and(overlay_mask_gt_thres, brain_mask).sum())
print(f"Covered voxels (>0): {cov0_vox:,d}")
print(f"Covered voxels (>thr): {covthres_vox:,d}")

# %%
