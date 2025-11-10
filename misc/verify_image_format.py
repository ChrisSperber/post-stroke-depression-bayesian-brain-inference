"""Check nii images in a folder to have uniform size & spacing and expected values.

The script is intended for explorative pre-analysis and is not part of the main analysis pipeline.

Requirements: All data are extracted into DATA_DIR with the modality in the name of each parent
    folder. (E.g. DATADIR / "Lesions" / lesion123.nii.gz)
"""

# %%
import numpy as np
from nibabel.nifti1 import Nifti1Image

from depression_mapping_tools.config import (
    DATA_DIR,
    DISCONNECTION_MAPS,
    IMAGE_TYPES,
    LESION,
    LESION_NETWORK,
)
from depression_mapping_tools.utils import load_nifti

NIFTI_EXTENSION = ".nii"

EXPECTED_LESION_VALUES = {0.0, 1.0}

# %%
files = [file for file in DATA_DIR.rglob("*") if file.is_file()]

print("Starting comparison")

for image_type in IMAGE_TYPES:

    nifti_list = [
        path
        for path in files
        if image_type in str(path) and NIFTI_EXTENSION in str(path)
    ]

    print(f"Comparing image type {image_type} for {len(nifti_list)} images")

    reference_image: Nifti1Image = load_nifti(
        nifti_list[0]
    )  # load first image as reference
    reference_header = reference_image.header
    reference_dim = reference_header["dim"]
    reference_pixdim = reference_header["pixdim"]
    reference_datatype = reference_header["data_type"]
    reference_affine = reference_image.affine

    for nifti_path in nifti_list:
        image = load_nifti(nifti_path)
        header = image.header

        if not np.array_equal(header["dim"], reference_dim):
            deviation_value = header["dim"]
            print(f"Deviating dim for {nifti_path};{deviation_value}")

        if not np.array_equal(header["pixdim"], reference_pixdim):
            deviation_value = header["pixdim"]
            print(f"Deviating pixdim for {nifti_path};{deviation_value}")

        if header["data_type"] != reference_datatype:
            deviation_value = header["data_type"]
            print(f"Deviating data_type for {nifti_path};{deviation_value}")

        if not np.allclose(reference_affine, image.affine, atol=1e-5):  # type: ignore
            print(f"Deviating affine for {nifti_path}")

        # verify that image only takes expected values
        image_arr = image.get_fdata()

        if image_type == LESION:
            unique_values = np.unique(image_arr).tolist()

            expected_found = {
                v
                for v in unique_values  # pyright: ignore[reportGeneralTypeIssues]
                if any(np.isclose(v, ev) for ev in EXPECTED_LESION_VALUES)
            }
            unexpected_values = {
                v
                for v in unique_values  # pyright: ignore[reportGeneralTypeIssues]
                if not any(np.isclose(v, ev) for ev in EXPECTED_LESION_VALUES)
            }

            if len(expected_found) != len(EXPECTED_LESION_VALUES):
                print(f"Lesion {nifti_path} only contains values {expected_found}")
            if unexpected_values:
                print(f"Lesion {nifti_path} contains values {unexpected_values}")

        if image_type == LESION_NETWORK:
            # values are Fisher transformed (artanh), hence tanh is required to re-convert to
            # Pearson R
            if np.tanh(np.max(image_arr)) > 1:
                print(f"LNM {nifti_path} contains values >1")
            if np.tanh(np.min(image_arr)) < -1:
                print(f"LNM {nifti_path} contains values >-1")

        if image_type == DISCONNECTION_MAPS:
            # voxel-wise disconnection probability
            if np.max(image_arr) > 1:
                print(f"DiscMap {nifti_path} contains values >1")
            if np.min(image_arr) < 0:
                print(f"DiscMap {nifti_path} contains values >0")


print("DONE")
# %%
