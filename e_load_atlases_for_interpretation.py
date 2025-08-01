"""Load atlasses to be used as reference to interpret topographical results.

Outputs:
    - Brain atlasses are loaded to Code/brain_atlasses
"""

# %%
import zipfile
from pathlib import Path

import requests
from nilearn.datasets import fetch_atlas_harvard_oxford

BRAIN_ATLAS_DIR = Path(__file__).parent / "brain_atlasses"

# %%
#####
# check availability of brain atlases and download if required
BRAIN_ATLAS_DIR.mkdir(exist_ok=True)

# Harvard-Oxford Atlas
# is included per default in the nilearn package
# the function downloads the atlases into a folder on the local machine outside of the repo folder
# a Bunch object is fetched including the NifTi1 Image in Bunch.maps and the label list in
# Bunch.labels

harvard_oxford_atlas_cort = fetch_atlas_harvard_oxford(
    atlas_name="cortl-maxprob-thr25-1mm", symmetric_split=True
)
harvard_oxford_atlas_subcort = fetch_atlas_harvard_oxford(
    atlas_name="sub-maxprob-thr25-1mm", symmetric_split=True
)

# %%
# Fibre definitions as used in the BCB Lab
fibres_dir = BRAIN_ATLAS_DIR / "BCB_Fibres"
download_dir = fibres_dir / "fibres_download"
extract_dir = fibres_dir / "fibres"

dropbox_download_link = "https://www.dropbox.com/scl/fi/u7xen7lqtymei39uzf1ti/Atlas_Rojkova.zip?rlkey=kp7rkevc6jclvhgo0r6z20nb4&dl=1"

fibres_dir.mkdir(exist_ok=True)

# Check if already downloaded
if download_dir.exists():
    print(f"Fibre definitions already downloaded: {download_dir}")
else:
    print("Downloading transmitter maps pt1")
    response = requests.get(dropbox_download_link, timeout=10)
    download_dir.write_bytes(response.content)
    print(f"Downloaded to: {download_dir}")

# check if already extracted
if extract_dir.exists():
    print(f"Fibre definitions already extracted: {extract_dir}")
else:
    print("Extracting fibre maps data")
    with zipfile.ZipFile(download_dir, "r") as zf:
        zf.extractall(extract_dir)

# %%
# Neurotransmitter maps as defined by Alves et al., 2025
# https://doi.org/10.1038/s41467-025-57680-2
transmitter_maps_link_1 = "https://neurovault.org/collections/15237/download"
transmitter_maps_link_2 = "https://neurovault.org/collections/17228/download"
download_dir = BRAIN_ATLAS_DIR / "neurotransmitter_maps"
zip_path_1 = download_dir / "Neurotransmitter_Maps_1.zip"
zip_path_2 = download_dir / "Neurotransmitter_Maps_2.zip"
extract_dir = download_dir / "neurotransmitter_maps_extracted"

# Check if already downloaded and extracted
if extract_dir.exists():
    print(f"Already extracted: {extract_dir}")
else:
    download_dir.mkdir(parents=True, exist_ok=True)
    if not zip_path_1.exists():
        print("Downloading transmitter maps pt1")
        response = requests.get(transmitter_maps_link_1, timeout=10)
        zip_path_1.write_bytes(response.content)
        print(f"Downloaded to: {zip_path_1}")
    if not zip_path_2.exists():
        print("Downloading transmitter maps pt2")
        response = requests.get(transmitter_maps_link_2, timeout=10)
        zip_path_2.write_bytes(response.content)
        print(f"Downloaded to: {zip_path_2}")
    print("Extracting transmitter data")
    with zipfile.ZipFile(zip_path_1, "r") as zf:
        zf.extractall(extract_dir)
    with zipfile.ZipFile(zip_path_2, "r") as zf:
        zf.extractall(extract_dir)

# %%
