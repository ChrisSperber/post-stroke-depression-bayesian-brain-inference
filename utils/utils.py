"""Utility functions and objects."""

# %%
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from nibabel import Nifti1Image
from tqdm import tqdm

PLACEHOLDER_FILE_NOT_EXIST = "None"

# define threshold to bin BF maps into strength of evidence categories; here chosen according to
# https://doi.org/10.3758/s13423-017-1323-7
BF_BINNING_THRESHOLDS_HO = [1 / 3, 1 / 10, 1 / 30, 1 / 100]
BF_BINNING_THRESHOLDS_H1 = [3, 10, 30, 100]


class BinnedBFMap(NamedTuple):
    """Named Tuple for binned BF maps.

    Stores voxelwise Bayes Factor maps for three hypotheses:
    H1 (evidence for effect), No evidence, H0 (evidence against effect).
    """

    bf_map_h1: np.ndarray
    bf_map_noev: np.ndarray
    bf_map_h0: np.ndarray


class CorrelationFormat(Enum):
    """Enum to define transform of Pearson Rs in LNM analysis."""

    ARTANH_PEARSON = "artanh_pearson_r"  # Fisher transformed; original sent data
    NONTRANSFORMED_PEARSON = "pearson_r"  # Original r value
    ATANH_PEARSON = "atanh_pearson_r"  # inversed Fisher transform atanh(x)
    POWER_TRANSFORM = "power_transformed_pearson_r"


class DisconnectionFormat(Enum):
    """Enum to define disconnection format as binary or continuous."""

    BINARY = "binarised_disconnection"  # binarised at chosen threshold
    CONTINUOUS = (
        "continuous_disconnection"  # original continuous scores (disconn. probability)
    )


def power_transform(x: float, beta: float = 0.5):
    """Power-law transform of value x with compression factor beta.

    Args:
        x (_type_): Input value.
        beta (_type_): Power, i.e. compression factor of function.

    Returns:Transformed value.

    """
    return np.sign(x) * np.abs(x) ** beta


def find_unique_path(paths: list[Path], str1: str, str2: str) -> str:
    """Find a unique filepath in a list containing 2 strings.

    Args:
        paths: List of path objects.
        str1: Substring to match 1
        str2: Substring to match 2

    Raises:
        ValueError: More than 1 matching file exists.

    Returns:
        Either file path or, if non-existent, placeholder string.

    """
    matches = [p for p in paths if str1 in str(p) and str2 in str(p)]

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise ValueError(f"Multiple matches found for {str1} and {str2}")
    else:
        return PLACEHOLDER_FILE_NOT_EXIST


def compare_image_affine_and_shape(
    reference_image: Nifti1Image, tested_image: Nifti1Image
) -> bool:
    """Check if affine and shape of an image match the reference image.

    Args:
        reference_image (Nifti1Image): Nifti1image in reference space.
        tested_image (Nifti1Image): Nifti1image to be tested against reference.

    Returns:
        bool: True if same, else False.

    """
    if not np.array_equal(reference_image.affine, tested_image.affine):
        return False
    elif reference_image.header.get_zooms()[:3] != tested_image.header.get_zooms()[:3]:
        return False
    elif reference_image.shape != tested_image.shape:
        return False
    else:
        return True


def compute_voxelwise_bf_via_glm(  # noqa: C901
    voxel_values: np.ndarray,
    target_var: np.ndarray,
    minimum_analysis_threshold: int | None,
    covariates: None | np.ndarray = None,
) -> float:
    """Compute the Bayes Factor for a voxel-wise value to be associated with a target variable.

    The Bayes Factor is approximated via the BICs of General Linear Models as described in
    Wagenmakers, E. J. (2007). A practical solution to the pervasive problems of p values.
    Psychonomic bulletin & review, 14(5), 779-804

    Args:
        voxel_values (np.array): Voxel-wise values (binary/continuous)
        target_var (np.array): Continuous target variable
        minimum_analysis_threshold: Minimum of lesions per voxel to be analysed.
            Does only apply to binary images values
        covariates (None | np.array, optional): Covariates. Defaults to None.

    Raises:
        ValueError: Height mismatch/unexpected size in input arrays.

    Returns:
        float: Bayes Factor

    """
    if voxel_values.shape[0] != target_var.shape[0]:
        raise ValueError(
            "Array heights do not match between voxel values and target variable."
        )
    if covariates:
        if voxel_values.shape[0] != covariates.shape[0]:
            raise ValueError(
                "Array heights do not match between voxel values and covariates."
            )
    if voxel_values.ndim != 1:
        raise ValueError("Voxel values are not a 1D array.")
    if target_var.ndim != 1:
        raise ValueError("Target valuess are not a 1D array.")

    # if number of 1s is below the analysis minimum analysis threshold, skip analysis
    if minimum_analysis_threshold:
        if np.count_nonzero(voxel_values == 1) < minimum_analysis_threshold:
            return 0.0
    # if no threshold given, still exclude voxels with only 0s (as in LNMs)
    elif np.count_nonzero(voxel_values) == 0:
        return 0.0

    df = pd.DataFrame({"voxel": voxel_values, "target": target_var})

    # Add covariates if they exist
    if covariates is not None:
        # If it's a 1D array, reshape to 2D (n_samples, 1)
        if covariates.ndim == 1:
            covariates = covariates[:, np.newaxis]

        n_covs = covariates.shape[1]
        for i in range(n_covs):
            df[f"cov{i}"] = covariates[:, i]

    # compute models
    x0 = sm.add_constant(df.drop(columns=["voxel", "target"]))
    x1 = sm.add_constant(df.drop(columns=["target"]))

    y = df["target"]

    model0 = sm.GLM(y, x0, family=sm.families.Gaussian()).fit()
    model1 = sm.GLM(y, x1, family=sm.families.Gaussian()).fit()

    bic0 = model0.bic
    bic1 = model1.bic

    log_bf = 0.5 * (bic0 - bic1)
    return np.exp(log_bf)


def run_voxelwise_bf_map(
    image_data_4d: np.ndarray,
    target_var: np.ndarray,
    minimum_analysis_threshold: int | None,
    covariates: None | np.ndarray = None,
    n_jobs: int = -1,
) -> np.ndarray:
    """Perform a parallelised mapping of the Bayes Factor on 4D imaging data.

    Args:
        image_data_4d (np.ndarray): 4D array of imaging data with size (n Subject, x, y, z)
        target_var (np.ndarray): 1D array with target variable
        minimum_analysis_threshold (int, optional): Minimum of lesions per voxel to be analysed.
            Does only apply to binary images values
        covariates (None | np.ndarray, optional): Array with covariates. Defaults to None.
        n_jobs (int, optional): Workers. Defaults to -1.

    Returns:
        np.ndarray: Array with map of Bayes Factors.

    """
    n_subjects, x_dim, y_dim, z_dim = image_data_4d.shape
    output_bf_map = np.full((x_dim, y_dim, z_dim), 0.0, dtype=np.float32)

    # Flatten voxel indices to loop over
    voxel_indices = [
        (x, y, z) for x in range(x_dim) for y in range(y_dim) for z in range(z_dim)
    ]

    def process_voxel(x, y, z):
        voxel_values = image_data_4d[:, x, y, z]
        try:
            bf = compute_voxelwise_bf_via_glm(
                voxel_values=voxel_values,
                target_var=target_var,
                minimum_analysis_threshold=minimum_analysis_threshold,
                covariates=covariates,
            )
            return (x, y, z, bf)
        except Exception:
            return (x, y, z, np.nan)

    # Run in parallel with progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_voxel)(x, y, z)
        for x, y, z in tqdm(voxel_indices, desc="Computing voxelwise BFs")
    )

    # Fill result map
    for x, y, z, bf in results:
        output_bf_map[x, y, z] = bf

    return output_bf_map


def bin_bf_map(bf_map: np.ndarray) -> BinnedBFMap:
    """Bin Bayes Factors in an array into evidence categories.

    Categories follow typical conventions as laid down e.g. in
    https://doi.org/10.3758/s13423-017-1323-7
    H1 and H0 maps are graded into 4 levels
    (for h1: 3,10,30,100 correspond to 1,2,3,4, for h0 inverted to 1/3, 1/10,...)

    Args:
        bf_map (np.ndarray): 3D array of Bayes Factors.

    Returns:
        BinnedBFMap: Named Tuple containing maps for h1, no evidence, and h0.

    """
    # no evidence
    map_no_ev = (bf_map >= BF_BINNING_THRESHOLDS_HO[0]) & (
        bf_map < BF_BINNING_THRESHOLDS_H1[0]
    )
    map_no_ev = map_no_ev.astype(np.uint8)
    # evidence h0
    map_h0_1 = (bf_map < BF_BINNING_THRESHOLDS_HO[0]) & (bf_map > 0)
    map_h0_1 = map_h0_1.astype(np.uint8)
    map_h0_2 = (bf_map < BF_BINNING_THRESHOLDS_HO[1]) & (bf_map > 0)
    map_h0_2 = map_h0_2.astype(np.uint8)
    map_h0_3 = (bf_map < BF_BINNING_THRESHOLDS_HO[2]) & (bf_map > 0)
    map_h0_3 = map_h0_3.astype(np.uint8)
    map_h0_4 = (bf_map < BF_BINNING_THRESHOLDS_HO[3]) & (bf_map > 0)
    map_h0_4 = map_h0_4.astype(np.uint8)
    map_h0_graded = map_h0_1 + map_h0_2 + map_h0_3 + map_h0_4
    map_h0_graded = map_h0_graded.astype(np.uint8)
    # evidence h1
    map_h1_1 = (bf_map >= BF_BINNING_THRESHOLDS_H1[0]).astype(np.uint8)
    map_h1_2 = (bf_map >= BF_BINNING_THRESHOLDS_H1[1]).astype(np.uint8)
    map_h1_3 = (bf_map >= BF_BINNING_THRESHOLDS_H1[2]).astype(np.uint8)
    map_h1_4 = (bf_map >= BF_BINNING_THRESHOLDS_H1[3]).astype(np.uint8)
    map_h1_graded = map_h1_1 + map_h1_2 + map_h1_3 + map_h1_4
    map_h1_graded = map_h1_graded.astype(np.uint8)

    return BinnedBFMap(
        bf_map_h1=map_h1_graded, bf_map_noev=map_no_ev, bf_map_h0=map_h0_graded
    )


# %%
