"""Paths and constants."""

from pathlib import Path

DATA_DIR = Path(__file__).parents[3] / "Data"
MASTER_FILE_CSV = DATA_DIR / "Masterfile_Behavioural_Scores_PSD_202511.xlsx"

LESION = "Lesions"
LESION_NETWORK = "LNM"
DISCONNECTION_MAPS = "SDSM"
IMAGE_TYPES = [LESION, LESION_NETWORK, DISCONNECTION_MAPS]
