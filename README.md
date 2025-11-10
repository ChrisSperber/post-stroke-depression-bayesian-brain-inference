# Bayesian Brain Mapping Scripts and Results for Connectome Mapping in Poststroke Depression

This repository contains the Python analysis code and summary outputs for Bayesian Lesion Deficit Inference (BLDI) with Lesion Segmentations and Connectomic data in Poststroke Depression.

> ⚠️ **Note**: This repository does **not** include raw data due to privacy/ethics restrictions.
---

## Repository Contents

| Folder/File                       | Description                                |
|-----------------------------------|--------------------------------------------|
| `*.py files`                      | Main analysis scripts using Python         |
| `misc/`                           | Additional scripts and files for exploration and visualisation (not part of main pipeline) |
| `BLDI_Outputs/`                   | Results of BLDI: .nii.gz and .txt logs     |
| `src/depression_mapping_tools`    | local package with functions and classes    |
| `a_collect_image_data.csv`        | Study ID list and exclusion info           |
| `requirements.txt`                | Python dependencies (from `venv`)          |
| `LICENSE`                         | MIT License                                |
---

## Reproducing the Analysis

This project was developed and run using Python 3.12.9 in a local `venv` environment.

### 1. Clone the repository
```bash
git clone https://github.com/ChrisSperber/post-stroke-depression-bayesian-brain-inference
cd post-stroke-depression-bayesian-brain-inference
```
### 2. Set up environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
# install local utils package in editable mode
pip install -e .
```
### 4. Run Analysis

The main analysis scripts are sequentially ordered with alphabetic prefixes. See docstrings for further information.
The code is intended to load large datasets of >2000 images, which was feasible with a 16GB RAM system, but required additional steps for float-precision data at 1mm³ isotropic resolution (see d_bayesian_inference_discmap.py). Systems with less RAM may require further changes to the code to be able to handle the data.

---

## Details on methods
This repository adapts [Bayesian Lesion Deficit Inference](https://pubmed.ncbi.nlm.nih.gov/36914109/), previously written in R with the [BayesFactor Package](https://cran.r-project.org/web/packages/BayesFactor/vignettes/manual.html), to Python.
The original implementation with the BayesFactor package required processing times that grew exponentially with sample size, and were already at 2-3 hours with sample sizes of 300 subjects [(see results section here)](https://pubmed.ncbi.nlm.nih.gov/36914109/). As the code was intended to be run on a large dataset on >2000 subjects, Bayes Factors were now computed using an [approximation](https://link.springer.com/article/10.3758/BF03194105) using the Bayesian Information criteria (BIC) of general linear models, as well as multi-kernel support.
As in the [original publication](https://pubmed.ncbi.nlm.nih.gov/36914109/), voxel-wise Bayes Factors indicating an association of the voxel's imaging data and a target variable (here: depression severity) are computed via general linear models.

### Analysis Outputs
Outputs are created in /BLDI_Outputs and include, for each analysis
- a Bayes Factor map containing the BF<sub>10</sub> BF_full_{modality}_{timestamp}.nii.gz, whereas modality gives the image type (lesion/lesion network mapoing[lnm]/disconnection mapping[sdsm])
- binned Bayes Factor maps for better visualisation of evidence categories BF_{evidence}_{modality}_{timestamp}.nii.gz, whereas evidence can be "no evidence" (noev), "evidence for h1" (h1), or "evidence for h0" (h0). Evidence for h1|h0 is graded into 1 - BF>3|BF<1/3; 2 - BF>10|BF<1/10 ; 3 - BF>30|BF<1/30; 4 - BF>100|BF<1/100
- a .txt file with meta-information on the analysis

---
### Reusability
Feel free to reuse and adapt the analysis code for similar datasets. Main functions are contained in utils/utils.py. The current code assumes all images to be located in the same reference space (like MNI space) and unified in shape and resolution.

### References
[Bayesian lesion-deficit inference with Bayes factor mapping: Key advantages, limitations, and a toolbox](https://doi.org/10.1016/j.neuroimage.2023.120008)

### License
This project is licensed under the MIT License — see [Project License](LICENSE) for details.

