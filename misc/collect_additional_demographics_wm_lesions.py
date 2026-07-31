"""Compute additional demographics for WM lesion patients that were excluded from the LNM.

For the lesion network mapping analysis, patients with pure white matter lesions were excluded as
no networks were computed for white matter damage. The current script compares the demographics of
those excluded vs. the remaining sample.

Outputs:
    - a yml with basic whole-sample information (N, aetiology, age, sex)
"""

# %%
from pathlib import Path

import pandas as pd
import yaml

from depression_mapping_tools.config import PLACEHOLDER_MISSING_VALUE
from depression_mapping_tools.utils import (
    DEPRESSION_GT_CUTOFFS,
    Cols,
)

COHORT = Cols.COHORT
DEPRESSION_BINARY = "DepressionBinary"
VARIABLE = "Variable"
STAT = "Stat"
VALUE = "Value"

# Cohort Names
IOWA = "Iowa"
KOREA = "Korea"
LEIPZIG = "Leipzig"
BORDEAUX = "Bordeaux"
# Korean sub-cohorts
HALLYM = "Hallym"
BUNDANG = "Bundang"

cohorts = [IOWA, KOREA, LEIPZIG, BORDEAUX]

DEPRESSION_MEASURE_MAP = {
    IOWA: [Cols.BDI_II],
    KOREA: [Cols.GDS15, Cols.GDS30],
    LEIPZIG: [Cols.HADS],
    BORDEAUX: [Cols.HADS],
}

DEPRESSION_SCORE_COLS = [Cols.GDS15, Cols.GDS30, Cols.BDI_II, Cols.HADS]

# %%
data = pd.read_csv(Path(__file__).parents[1] / "a_collect_image_data.csv")
data = data[data[Cols.EXCLUDED] == 0]

# replace Korean subcohort names with meta-cohort name to create a single summary
data[Cols.COHORT] = data[Cols.COHORT].replace([HALLYM, BUNDANG], KOREA)

# derive binary depression classification based on cutoffs
depression_classification_list = []
for _, row in data.iterrows():
    mask = row[DEPRESSION_SCORE_COLS] != PLACEHOLDER_MISSING_VALUE
    col_name = row[DEPRESSION_SCORE_COLS][mask].index[0]
    depression_value = int(row[DEPRESSION_SCORE_COLS][mask].iloc[0])

    relevant_cutoff = DEPRESSION_GT_CUTOFFS.get(col_name)
    if relevant_cutoff:
        depression_classification_list.append(int(depression_value > relevant_cutoff))
    else:
        raise ValueError("Relevant depression cutoff was not derived!")
data[DEPRESSION_BINARY] = depression_classification_list


# %%
# divide into subsamples
data_included_in_lnm = data.copy()
data_included_in_lnm = data_included_in_lnm[
    ~data_included_in_lnm[Cols.PATH_LNM_IMAGE].isna()
]
data_excluded_in_lnm = data.copy()
data_excluded_in_lnm = data_excluded_in_lnm[
    data_excluded_in_lnm[Cols.PATH_LNM_IMAGE].isna()
]


# %%
def summarize_sample(df: pd.DataFrame, name: str) -> dict:  # noqa: PLR0915
    """Print and return demographic summary statistics for one sample."""
    print(f"\n{'=' * 60}")
    print(name)
    print("=" * 60)

    n_total = len(df)
    print(f"Total N: {n_total}")

    # ---------- Age ----------
    age = pd.to_numeric(df[Cols.AGE], errors="coerce")
    age_mean = round(age.mean(), 2)
    age_sd = round(age.std(), 2)
    age_min = round(age.min(), 2)
    age_max = round(age.max(), 2)

    print("------\nAge")
    print(f"Mean: {age_mean}")
    print(f"SD: {age_sd}")
    print(f"Range: {age_min}-{age_max}")

    # ---------- Sex ----------
    print("------\nSex")
    n_male = int((df[Cols.SEX] == "Male").sum())
    n_female = int((df[Cols.SEX] == "Female").sum())

    print(f"Male: {n_male}, {n_male / n_total * 100:.2f}%")
    print(f"Female: {n_female}, {n_female / n_total * 100:.2f}%")

    # ---------- Aetiology ----------
    print("------\nAetiology")
    n_ischaemia = int((df[Cols.AETIOLOGY] == "Ischaemic_Stroke").sum())
    n_icb = int((df[Cols.AETIOLOGY] == "ICB").sum())

    print(f"Ischaemia: {n_ischaemia}, {n_ischaemia / n_total * 100:.2f}%")
    print(f"ICB: {n_icb}, {n_icb / n_total * 100:.2f}%")

    # ---------- Lesion volume ----------
    lesion_vol = pd.to_numeric(df[Cols.LESION_VOLUME], errors="coerce")
    median_lesvol = round(lesion_vol.median(), 1)
    q1 = round(lesion_vol.quantile(0.25), 1)
    q3 = round(lesion_vol.quantile(0.75), 1)

    print("------\nLesion Volume")
    print(f"Median: {median_lesvol}")
    print(f"IQR: {q1} - {q3}")

    # ---------- Depression ----------
    n_dep = df[DEPRESSION_BINARY].sum()
    dep_percent = round(n_dep / n_total * 100, 2)

    print("------\nDepression")
    print(f"{n_dep}/{n_total} ({dep_percent}%)")

    # ---------- NIHSS ----------
    nihss = (
        df[Cols.NIHSS_ON_ADMISSION]
        .replace(PLACEHOLDER_MISSING_VALUE, pd.NA)
        .pipe(pd.to_numeric, errors="coerce")
    )

    median_nihss = round(nihss.median(), 1)
    q1_nihss = round(nihss.quantile(0.25), 1)
    q3_nihss = round(nihss.quantile(0.75), 1)

    print("------\nNIHSS on admission")
    print(f"N available: {nihss.notna().sum()}")
    print(f"Median: {median_nihss}")
    print(f"IQR: {q1_nihss} - {q3_nihss}")

    # ---------- Follow-up interval ----------
    followup = (
        df[Cols.DAYS_ONSET_TO_FOLLOWUP]
        .replace(PLACEHOLDER_MISSING_VALUE, pd.NA)
        .pipe(pd.to_numeric, errors="coerce")
    )

    median_followup = round(followup.median(), 1)
    q1_followup = round(followup.quantile(0.25), 1)
    q3_followup = round(followup.quantile(0.75), 1)

    print("------\nDays onset to follow-up")
    print(f"N available: {followup.notna().sum()}")
    print(f"Median: {median_followup}")
    print(f"IQR: {q1_followup} - {q3_followup}")

    return {
        "total_n": int(n_total),
        "age": {
            "mean": float(age_mean),
            "sd": float(age_sd),
            "min": float(age_min),
            "max": float(age_max),
        },
        "depression": {
            "n": int(n_dep),
            "percent": float(dep_percent),
        },
        "sex": {
            "male": {
                "n": int(n_male),
                "percent": round(n_male / n_total * 100, 2),
            },
            "female": {
                "n": int(n_female),
                "percent": round(n_female / n_total * 100, 2),
            },
        },
        "aetiology": {
            "ischaemia": {
                "n": int(n_ischaemia),
                "percent": round(n_ischaemia / n_total * 100, 2),
            },
            "icb": {
                "n": int(n_icb),
                "percent": round(n_icb / n_total * 100, 2),
            },
        },
        "lesion_volume": {
            "median": float(median_lesvol),
            "iqr": [float(q1), float(q3)],
        },
        "nihss_on_admission": {
            "n": int(nihss.notna().sum()),
            "median": float(median_nihss),
            "iqr": f"{q1_nihss} - {q3_nihss}",
        },
        "days_onset_to_followup": {
            "n": int(followup.notna().sum()),
            "median": float(median_followup),
            "iqr": f"{q1_followup} - {q3_followup}",
        },
    }


# %%
summary = {
    "included_in_LNM": summarize_sample(data_included_in_lnm, "Included in LNM"),
    "excluded_in_LNM": summarize_sample(data_excluded_in_lnm, "Excluded from LNM"),
}

out_path = Path(__file__).with_suffix(".yml")
with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(summary, f, sort_keys=False)

# %%
