"""Paths and constants."""

from pathlib import Path

DATA_DIR = Path(__file__).parents[3] / "Data"
MASTER_FILE_EXCEL = DATA_DIR / "Masterfile_Behavioural_Scores_PSD_202511.xlsx"

LESION = "Lesions"
LESION_NETWORK = "LNM"
DISCONNECTION_MAPS = "SDSM"
IMAGE_TYPES = [LESION, LESION_NETWORK, DISCONNECTION_MAPS]

PLACEHOLDER_FILE_NOT_EXIST = "None"
PLACEHOLDER_MISSING_VALUE = "Not_available"

BLDI_OUTPUT_DIR_PARENT = Path(__file__).parents[2] / "BLDI_OUTPUTS"
# Define the minimum amount of lesions per voxel to be included in the analysis
MIN_LESION_ANALYSIS_THRESHOLD = 10
# ... also for binary Disconnection Maps
MIN_DISCONNECTION_ANALYSIS_THRESHOLD = 10

BINARY_THRESHOLD_DISCMAP = (
    0.6  # if DisconnectionFormat.BINARY, all values >= this value are set to 1 else 0
)
