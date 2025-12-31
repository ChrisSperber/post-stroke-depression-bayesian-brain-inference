"""Compute demographics for manuscript text and tables.

Detailed demographics are collected independently for each cohort.

Outputs:
    - a csv with demographic data per cohort
    - a yml with basic whole-sample information (N, aetiology, age, sex)
"""

# %%
from pathlib import Path

import pandas as pd
import yaml

from depression_mapping_tools.config import PLACEHOLDER_MISSING_VALUE
from depression_mapping_tools.utils import Cols, all_missing_or_placeholder

COHORT = Cols.COHORT
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

# %%
data = pd.read_csv(Path(__file__).parent / "a_collect_image_data.csv")
data = data[data[Cols.EXCLUDED] == 0]

# replace Korean subcohort names with meta-cohort name to create a single summary
data[Cols.COHORT] = data[Cols.COHORT].replace([HALLYM, BUNDANG], KOREA)

# print general whole sample statistics to terminal
n_total = len(data)
print(f"Total N: {n_total}")

age = pd.to_numeric(data[Cols.AGE], errors="coerce")
mean_val = round(age.mean(), 2)
sd_val = round(age.std(), 2)
min_val = round(age.min(), 2)
max_val = round(age.max(), 2)

lesion_vol = data[Cols.LESION_VOLUME]
median_lesvol = round(lesion_vol.median(), 1)
iqr_lesvol = [
    float(round(lesion_vol.quantile(0.25), 1)),
    float(round(lesion_vol.quantile(0.75), 1)),
]

print("------\nAge")
print(f"Mean: {mean_val}")
print(f"Std: {sd_val}")
print(f"Range: {min_val}-{max_val}")

print("------\nSex")
n_male = sum(data[Cols.SEX] == "Male")
n_female = sum(data[Cols.SEX] == "Female")

print(f"Male: {n_male}, {round(n_male/n_total*100,2)}%")
print(f"Female: {n_female}, {round(n_female/n_total*100,2)}%")

print("------\nAetiology")
n_ischaemia = sum(data[Cols.AETIOLOGY] == "Ischaemic_Stroke")
n_icb = sum(data[Cols.AETIOLOGY] == "ICB")

print(f"Ischmaemia: {n_ischaemia}, {round(n_ischaemia/n_total*100,2)}%")
print(f"ICB: {n_icb}, {round(n_icb/n_total*100,2)}%")

print("------\nLesion Volume")
print(f"Median: {median_lesvol}")
print(f"IQR: {iqr_lesvol}")

summary = {
    "total_n": int(n_total),
    "age": {
        "mean": float(mean_val),
        "std": float(sd_val),
        "min": float(min_val),
        "max": float(max_val),
    },
    "sex": {
        "male": {"n": int(n_male), "percent": float(round(n_male / n_total * 100, 2))},
        "female": {
            "n": int(n_female),
            "percent": float(round(n_female / n_total * 100, 2)),
        },
    },
    "aetiology": {
        "ischaemia": {
            "n": int(n_ischaemia),
            "percent": float(round(n_ischaemia / n_total * 100, 2)),
        },
        "icb": {"n": int(n_icb), "percent": float(round(n_icb / n_total * 100, 2))},
    },
    "lesion_volume": {
        "median": float(median_lesvol),
        "iqr": f"{iqr_lesvol[0]} - {iqr_lesvol[1]}",
    },
}

out_path = Path(__file__).with_suffix(".yml")
with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(summary, f, sort_keys=False)

# %%
statistical_results_list = []

for cohort in cohorts:
    print(f"Analyzing Cohort {cohort}")
    cohort_df = data.loc[data[Cols.COHORT] == cohort, :]
    print(f"Cohort sample size: {len(cohort_df)}")

    statistical_results_list.append(
        {COHORT: cohort, VARIABLE: "Sample_Size", STAT: "N", VALUE: len(cohort_df)}
    )

    # Aetiology
    n_ischaemia = sum(cohort_df[Cols.AETIOLOGY] == "Ischaemic_Stroke")
    n_icb = sum(cohort_df[Cols.AETIOLOGY] == "ICB")
    aetiology_str = f"Ischemia:{n_ischaemia}, ICB:{n_icb}"

    statistical_results_list.append(
        {
            COHORT: cohort,
            VARIABLE: Cols.AETIOLOGY,
            STAT: "N",
            VALUE: aetiology_str,
        }
    )

    # Age
    age = pd.to_numeric(cohort_df[Cols.AGE], errors="coerce")
    mean_val = round(age.mean(), 2)
    sd_val = round(age.std(), 2)
    min_val = round(age.min(), 2)
    max_val = round(age.max(), 2)

    age_str = f"Mean:{mean_val}, SD:{sd_val}, Range:{min_val}-{max_val} "

    statistical_results_list.append(
        {
            COHORT: cohort,
            VARIABLE: Cols.AGE,
            STAT: "Mean, SD, Range",
            VALUE: age_str,
        }
    )

    # Sex
    n_male = sum(cohort_df[Cols.SEX] == "Male")
    n_female = sum(cohort_df[Cols.SEX] == "Female")

    sex_str = f"Male:{n_male}, Female:{n_female}"

    statistical_results_list.append(
        {
            COHORT: cohort,
            VARIABLE: Cols.SEX,
            STAT: "N",
            VALUE: sex_str,
        }
    )

    # Handedness
    if all_missing_or_placeholder(cohort_df[Cols.HANDEDNESS]):
        statistical_results_list.append(
            {
                COHORT: cohort,
                VARIABLE: Cols.HANDEDNESS,
                STAT: "N",
                VALUE: PLACEHOLDER_MISSING_VALUE,
            }
        )
    else:
        n_righthanded = sum(cohort_df[Cols.HANDEDNESS] == "right")
        n_lefthanded = sum(cohort_df[Cols.HANDEDNESS] == "left")
        n_otherhanded = sum(cohort_df[Cols.HANDEDNESS] == "other")

        handedness_str = f"R:{n_righthanded}, L:{n_lefthanded}, Other:{n_otherhanded}"
        statistical_results_list.append(
            {
                COHORT: cohort,
                VARIABLE: Cols.HANDEDNESS,
                STAT: "N",
                VALUE: handedness_str,
            }
        )

    # Lesion Volume
    lesion_volume = cohort_df[Cols.LESION_VOLUME]
    lesion_volume_str = (
        f"Median:{lesion_volume.median()}, "
        f"IQR:[{lesion_volume.quantile(0.25)}, {lesion_volume.quantile(0.75)}]"
    )

    statistical_results_list.append(
        {
            COHORT: cohort,
            VARIABLE: Cols.LESION_VOLUME,
            STAT: "Median, IQR",
            VALUE: lesion_volume_str,
        }
    )

    # NIHSS on Admission
    if all_missing_or_placeholder(cohort_df[Cols.NIHSS_ON_ADMISSION]):
        statistical_results_list.append(
            {
                COHORT: cohort,
                VARIABLE: Cols.NIHSS_ON_ADMISSION,
                STAT: "Median, IQR",
                VALUE: PLACEHOLDER_MISSING_VALUE,
            }
        )
    else:
        nihss_admission = pd.to_numeric(
            cohort_df[Cols.NIHSS_ON_ADMISSION], errors="coerce"
        )
        nihss_str = (
            f"Median:{nihss_admission.median()}, "
            f"IQR:[{nihss_admission.quantile(0.25)}, {nihss_admission.quantile(0.75)}]"
        )
        statistical_results_list.append(
            {
                COHORT: cohort,
                VARIABLE: Cols.NIHSS_ON_ADMISSION,
                STAT: "Median, IQR",
                VALUE: nihss_str,
            }
        )
    # DaysToFollowup
    if all_missing_or_placeholder(cohort_df[Cols.DAYS_ONSET_TO_FOLLOWUP]):
        statistical_results_list.append(
            {
                COHORT: cohort,
                VARIABLE: Cols.DAYS_ONSET_TO_FOLLOWUP,
                STAT: "Median, IQR",
                VALUE: PLACEHOLDER_MISSING_VALUE,
            }
        )
    else:
        days_to_followup = (
            pd.to_numeric(cohort_df[Cols.DAYS_ONSET_TO_FOLLOWUP], errors="coerce")
            .round(0)
            .astype(int)
        )
        days_followup_str = (
            f"Median:{days_to_followup.median()}, "
            f"IQR:[{days_to_followup.quantile(0.25)}, {days_to_followup.quantile(0.75)}]"
        )
        statistical_results_list.append(
            {
                COHORT: cohort,
                VARIABLE: Cols.DAYS_ONSET_TO_FOLLOWUP,
                STAT: "Median, IQR",
                VALUE: days_followup_str,
            }
        )
    # Depression Measure

    depression_measures = DEPRESSION_MEASURE_MAP.get(cohort)
    if not depression_measures:
        msg = f"No depression measure found for cohort {cohort}"
        raise ValueError(msg)

    for depression_score in depression_measures:
        depr_scores_series = pd.to_numeric(cohort_df[depression_score], errors="coerce")
        depr_scores_series = depr_scores_series.dropna()

        depression_score_str = (
            f"N: {len(depr_scores_series)}, Median:{depr_scores_series.median()}, "
            f"IQR:[{depr_scores_series.quantile(0.25)}, {depr_scores_series.quantile(0.75)}]"
        )

        statistical_results_list.append(
            {
                COHORT: cohort,
                VARIABLE: "DepressionScore",
                STAT: f"{depression_score}: N, Median, IQR",
                VALUE: depression_score_str,
            }
        )

# %%
# store results
results = pd.DataFrame(statistical_results_list)
output_name = Path(__file__).with_suffix(".csv")
results.to_csv(output_name, index=False)

# %%
