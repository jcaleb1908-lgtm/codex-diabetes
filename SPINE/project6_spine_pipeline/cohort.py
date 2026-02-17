"""Cohort construction + analytic dataset extraction for Project 6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

try:
    from google.cloud import bigquery
except Exception:  # pragma: no cover - local shells may not include google sdk
    bigquery = Any  # type: ignore[assignment]

from .bq_utils import array_param, materialize_query, run_query, scalar_param


@dataclass
class CohortData:
    cohort_flow: pd.DataFrame
    analytic_df: pd.DataFrame
    post_index_bmi_df: pd.DataFrame


def _query_params(config: dict, concepts: dict) -> list[bigquery.ArrayQueryParameter | bigquery.ScalarQueryParameter]:
    return [
        array_param("metformin_names", "STRING", [x.lower() for x in concepts["METFORMIN_INGREDIENT_NAMES"]]),
        array_param("glp1_names", "STRING", [x.lower() for x in concepts["GLP1_INGREDIENT_NAMES"]]),
        array_param("t2dm_ancestor_concepts", "INT64", concepts["T2DM_ANCESTOR_CONCEPTS"]),
        array_param("dm_exclusion_ancestor_concepts", "INT64", concepts["DM_EXCLUSION_ANCESTOR_CONCEPTS"]),
        array_param("spine_concept_ids", "INT64", concepts["SPINE_OUTCOME_SNOMED_CONCEPT_IDS"]),
        array_param("bmi_measurement_concept_ids", "INT64", concepts["BMI_MEASUREMENT_CONCEPT_IDS"]),
        array_param("hba1c_measurement_concept_ids", "INT64", concepts["HBA1C_MEASUREMENT_CONCEPT_IDS"]),
        array_param(
            "outpatient_visit_ancestor_concept_ids",
            "INT64",
            concepts["OUTPATIENT_VISIT_ANCESTOR_CONCEPT_IDS"],
        ),
        scalar_param("adult_age_min", "INT64", int(config["adult_age_min"])),
        scalar_param("adult_age_max", "INT64", int(config["adult_age_max"])),
        scalar_param("lookback_days", "INT64", int(config["lookback_days"])),
        scalar_param("combo_window_days", "INT64", int(config["combo_window_days"])),
        scalar_param("outcome_window_days", "INT64", int(config["outcome_window_days"])),
        scalar_param("bmi_window_start_days", "INT64", int(config["bmi_baseline_window_start_days"])),
        scalar_param("bmi_window_end_days", "INT64", int(config["bmi_baseline_window_end_days"])),
        scalar_param("bmi_min", "FLOAT64", float(config["bmi_min"])),
        scalar_param("bmi_max", "FLOAT64", float(config["bmi_max"])),
    ]


def build_query_params(
    config: dict, concepts: dict
) -> list[bigquery.ArrayQueryParameter | bigquery.ScalarQueryParameter]:
    return _query_params(config, concepts)


def _core_ctes(dataset: str) -> str:
    return f"""
WITH
metformin_ingred AS (
  SELECT concept_id
  FROM `{dataset}.concept`
  WHERE vocabulary_id = 'RxNorm'
    AND concept_class_id = 'Ingredient'
    AND LOWER(concept_name) IN UNNEST(@metformin_names)
),
metformin_concepts AS (
  SELECT concept_id FROM metformin_ingred
  UNION DISTINCT
  SELECT ca.descendant_concept_id AS concept_id
  FROM `{dataset}.concept_ancestor` ca
  JOIN metformin_ingred mi
    ON ca.ancestor_concept_id = mi.concept_id
),

glp1_ingred AS (
  SELECT concept_id
  FROM `{dataset}.concept`
  WHERE vocabulary_id = 'RxNorm'
    AND concept_class_id = 'Ingredient'
    AND LOWER(concept_name) IN UNNEST(@glp1_names)
),
glp1_concepts AS (
  SELECT concept_id FROM glp1_ingred
  UNION DISTINCT
  SELECT ca.descendant_concept_id AS concept_id
  FROM `{dataset}.concept_ancestor` ca
  JOIN glp1_ingred gi
    ON ca.ancestor_concept_id = gi.concept_id
),

insulin_ingred AS (
  SELECT concept_id
  FROM `{dataset}.concept`
  WHERE vocabulary_id = 'RxNorm'
    AND concept_class_id = 'Ingredient'
    AND LOWER(concept_name) LIKE 'insulin%'
),
insulin_concepts AS (
  SELECT concept_id FROM insulin_ingred
  UNION DISTINCT
  SELECT ca.descendant_concept_id AS concept_id
  FROM `{dataset}.concept_ancestor` ca
  JOIN insulin_ingred ii
    ON ca.ancestor_concept_id = ii.concept_id
),

t2dm_concepts AS (
  SELECT DISTINCT descendant_concept_id AS concept_id
  FROM `{dataset}.concept_ancestor`
  WHERE ancestor_concept_id IN UNNEST(@t2dm_ancestor_concepts)
  UNION DISTINCT
  SELECT concept_id
  FROM `{dataset}.concept`
  WHERE concept_id IN UNNEST(@t2dm_ancestor_concepts)
),

dm_exclusion_concepts AS (
  SELECT DISTINCT descendant_concept_id AS concept_id
  FROM `{dataset}.concept_ancestor`
  WHERE ancestor_concept_id IN UNNEST(@dm_exclusion_ancestor_concepts)
  UNION DISTINCT
  SELECT concept_id
  FROM `{dataset}.concept`
  WHERE concept_id IN UNNEST(@dm_exclusion_ancestor_concepts)
),

spine_outcome_concepts AS (
  SELECT concept_id
  FROM `{dataset}.concept`
  WHERE concept_id IN UNNEST(@spine_concept_ids)
),

outpatient_visit_concepts AS (
  SELECT DISTINCT descendant_concept_id AS concept_id
  FROM `{dataset}.concept_ancestor`
  WHERE ancestor_concept_id IN UNNEST(@outpatient_visit_ancestor_concept_ids)
  UNION DISTINCT
  SELECT concept_id
  FROM `{dataset}.concept`
  WHERE concept_id IN UNNEST(@outpatient_visit_ancestor_concept_ids)
),

first_t2dm_dx AS (
  SELECT
    co.person_id,
    MIN(co.condition_start_date) AS first_t2dm_dx_date
  FROM `{dataset}.condition_occurrence` co
  JOIN t2dm_concepts t2
    ON co.condition_concept_id = t2.concept_id
  WHERE co.condition_start_date IS NOT NULL
  GROUP BY co.person_id
),

drug_starts AS (
  SELECT
    de.person_id,
    MIN(IF(mc.concept_id IS NOT NULL, de.drug_exposure_start_date, NULL)) AS metformin_start_date,
    MIN(IF(gc.concept_id IS NOT NULL, de.drug_exposure_start_date, NULL)) AS glp1_start_date
  FROM `{dataset}.drug_exposure` de
  LEFT JOIN metformin_concepts mc
    ON de.drug_concept_id = mc.concept_id
  LEFT JOIN glp1_concepts gc
    ON de.drug_concept_id = gc.concept_id
  WHERE de.drug_exposure_start_date IS NOT NULL
  GROUP BY de.person_id
),

exposure_assigned AS (
  SELECT
    ds.person_id,
    ds.metformin_start_date,
    ds.glp1_start_date,
    CASE
      WHEN ds.metformin_start_date IS NOT NULL
           AND ds.glp1_start_date IS NULL
        THEN 'metformin_only'
      WHEN ds.glp1_start_date IS NOT NULL
           AND ds.metformin_start_date IS NULL
        THEN 'glp1_only'
      WHEN ds.metformin_start_date IS NOT NULL
           AND ds.glp1_start_date IS NOT NULL
           AND ABS(DATE_DIFF(ds.metformin_start_date, ds.glp1_start_date, DAY)) <= @combo_window_days
        THEN 'combo'
      ELSE NULL
    END AS exposure_group,
    CASE
      WHEN ds.metformin_start_date IS NOT NULL
           AND ds.glp1_start_date IS NULL
        THEN ds.metformin_start_date
      WHEN ds.glp1_start_date IS NOT NULL
           AND ds.metformin_start_date IS NULL
        THEN ds.glp1_start_date
      WHEN ds.metformin_start_date IS NOT NULL
           AND ds.glp1_start_date IS NOT NULL
           AND ABS(DATE_DIFF(ds.metformin_start_date, ds.glp1_start_date, DAY)) <= @combo_window_days
        THEN LEAST(ds.metformin_start_date, ds.glp1_start_date)
      ELSE NULL
    END AS index_date
  FROM drug_starts ds
),

exposure_clean AS (
  SELECT *
  FROM exposure_assigned
  WHERE exposure_group IS NOT NULL
    AND index_date IS NOT NULL
),

t2dm_eligible AS (
  SELECT
    ec.person_id,
    ec.exposure_group,
    ec.index_date,
    ec.metformin_start_date,
    ec.glp1_start_date,
    t2.first_t2dm_dx_date
  FROM exposure_clean ec
  JOIN first_t2dm_dx t2
    ON t2.person_id = ec.person_id
   AND t2.first_t2dm_dx_date <= ec.index_date
),

adult_population AS (
  SELECT
    te.*,
    DATE(p.birth_datetime) AS birth_date,
    SAFE_DIVIDE(DATE_DIFF(te.index_date, DATE(p.birth_datetime), DAY), 365.25) AS age_at_index,
    p.sex_at_birth_concept_id,
    p.race_concept_id,
    p.ethnicity_concept_id
  FROM t2dm_eligible te
  JOIN `{dataset}.person` p
    ON p.person_id = te.person_id
  WHERE SAFE_DIVIDE(DATE_DIFF(te.index_date, DATE(p.birth_datetime), DAY), 365.25)
        BETWEEN @adult_age_min AND @adult_age_max
),

observation_eligible AS (
  SELECT
    ap.*,
    op.observation_period_start_date,
    op.observation_period_end_date,
    ROW_NUMBER() OVER (
      PARTITION BY ap.person_id
      ORDER BY op.observation_period_end_date DESC
    ) AS op_rn
  FROM adult_population ap
  JOIN `{dataset}.observation_period` op
    ON op.person_id = ap.person_id
   AND ap.index_date BETWEEN op.observation_period_start_date AND op.observation_period_end_date
   AND op.observation_period_start_date <= DATE_SUB(ap.index_date, INTERVAL @lookback_days DAY)
),

baseline_cohort AS (
  SELECT *
  FROM observation_eligible
  WHERE op_rn = 1
),

dm_exclusion_flag AS (
  SELECT DISTINCT b.person_id
  FROM baseline_cohort b
  JOIN `{dataset}.condition_occurrence` co
    ON co.person_id = b.person_id
   AND co.condition_start_date <= b.index_date
  JOIN dm_exclusion_concepts de
    ON co.condition_concept_id = de.concept_id
),

spine_dx AS (
  SELECT
    b.person_id,
    b.index_date,
    MIN(IF(co.condition_start_date <= b.index_date, co.condition_start_date, NULL)) AS spine_dx_on_or_before_index,
    MIN(IF(co.condition_start_date > b.index_date, co.condition_start_date, NULL)) AS first_spine_dx_after_index,
    MIN(IF(co.condition_start_date > b.index_date
           AND co.condition_start_date <= DATE_ADD(b.index_date, INTERVAL @outcome_window_days DAY),
           co.condition_start_date, NULL)) AS first_spine_dx_2y
  FROM baseline_cohort b
  LEFT JOIN `{dataset}.condition_occurrence` co
    ON co.person_id = b.person_id
  LEFT JOIN spine_outcome_concepts soc
    ON co.condition_concept_id = soc.concept_id
  WHERE soc.concept_id IS NOT NULL OR co.person_id IS NULL
  GROUP BY b.person_id, b.index_date
),

bmi_baseline_ranked AS (
  SELECT
    b.person_id,
    b.index_date,
    m.measurement_date,
    m.value_as_number AS bmi,
    ROW_NUMBER() OVER (
      PARTITION BY b.person_id, b.index_date
      ORDER BY
        CASE WHEN m.measurement_date <= b.index_date THEN 0 ELSE 1 END,
        ABS(DATE_DIFF(m.measurement_date, b.index_date, DAY)),
        m.measurement_date DESC
    ) AS rn
  FROM baseline_cohort b
  JOIN `{dataset}.measurement` m
    ON m.person_id = b.person_id
   AND m.measurement_concept_id IN UNNEST(@bmi_measurement_concept_ids)
   AND m.value_as_number BETWEEN @bmi_min AND @bmi_max
   AND m.measurement_date BETWEEN DATE_ADD(b.index_date, INTERVAL @bmi_window_start_days DAY)
                             AND DATE_ADD(b.index_date, INTERVAL @bmi_window_end_days DAY)
),

bmi_baseline AS (
  SELECT
    person_id,
    index_date,
    bmi,
    measurement_date AS bmi_measurement_date
  FROM bmi_baseline_ranked
  WHERE rn = 1
),

hba1c_raw AS (
  SELECT
    b.person_id,
    b.index_date,
    m.measurement_date,
    m.value_as_number AS hba1c_value
  FROM baseline_cohort b
  JOIN `{dataset}.measurement` m
    ON m.person_id = b.person_id
   AND m.measurement_concept_id IN UNNEST(@hba1c_measurement_concept_ids)
   AND m.value_as_number IS NOT NULL
   AND m.measurement_date BETWEEN DATE_SUB(b.index_date, INTERVAL @lookback_days DAY)
                             AND b.index_date
),

hba1c_baseline AS (
  SELECT
    person_id,
    index_date,
    ARRAY_AGG(hba1c_value ORDER BY measurement_date DESC LIMIT 1)[SAFE_OFFSET(0)] AS hba1c_recent,
    AVG(hba1c_value) AS hba1c_mean_year,
    COUNT(*) AS hba1c_measurements_baseline
  FROM hba1c_raw
  GROUP BY person_id, index_date
),

insulin_baseline AS (
  SELECT
    b.person_id,
    b.index_date,
    MAX(IF(ic.concept_id IS NOT NULL, 1, 0)) AS insulin_use_baseline
  FROM baseline_cohort b
  LEFT JOIN `{dataset}.drug_exposure` de
    ON de.person_id = b.person_id
   AND de.drug_exposure_start_date BETWEEN DATE_SUB(b.index_date, INTERVAL @lookback_days DAY)
                                      AND b.index_date
  LEFT JOIN insulin_concepts ic
    ON de.drug_concept_id = ic.concept_id
  GROUP BY b.person_id, b.index_date
),

complications_baseline AS (
  SELECT
    b.person_id,
    b.index_date,
    MAX(IF(LOWER(c.concept_name) LIKE '%diabet%neuropathy%', 1, 0)) AS neuropathy_baseline,
    MAX(IF(LOWER(c.concept_name) LIKE '%diabet%nephropathy%' OR LOWER(c.concept_name) LIKE '%diabetic kidney%', 1, 0)) AS nephropathy_baseline,
    MAX(IF(LOWER(c.concept_name) LIKE '%diabet%retinopathy%', 1, 0)) AS retinopathy_baseline,
    COUNT(DISTINCT co.condition_concept_id) AS baseline_condition_count
  FROM baseline_cohort b
  LEFT JOIN `{dataset}.condition_occurrence` co
    ON co.person_id = b.person_id
   AND co.condition_start_date BETWEEN DATE_SUB(b.index_date, INTERVAL @lookback_days DAY)
                                  AND b.index_date
  LEFT JOIN `{dataset}.concept` c
    ON c.concept_id = co.condition_concept_id
  GROUP BY b.person_id, b.index_date
),

utilization AS (
  SELECT
    b.person_id,
    b.index_date,
    COUNTIF(
      vo.visit_start_date BETWEEN DATE_SUB(b.index_date, INTERVAL @lookback_days DAY) AND b.index_date
      AND ovc.concept_id IS NOT NULL
    ) AS baseline_outpatient_visits,
    COUNTIF(
      vo.visit_start_date > b.index_date
      AND vo.visit_start_date <= DATE_ADD(b.index_date, INTERVAL @outcome_window_days DAY)
      AND ovc.concept_id IS NOT NULL
    ) AS followup_outpatient_visits_2y,
    COUNTIF(
      vo.visit_start_date BETWEEN DATE_SUB(b.index_date, INTERVAL @lookback_days DAY) AND b.index_date
      AND LOWER(sc.concept_name) LIKE '%endocrin%'
    ) AS baseline_endocrinology_visits,
    COUNTIF(
      vo.visit_start_date BETWEEN DATE_SUB(b.index_date, INTERVAL @lookback_days DAY) AND b.index_date
      AND LOWER(sc.concept_name) LIKE '%orthop%'
    ) AS baseline_orthopedics_visits
  FROM baseline_cohort b
  LEFT JOIN `{dataset}.visit_occurrence` vo
    ON vo.person_id = b.person_id
  LEFT JOIN outpatient_visit_concepts ovc
    ON vo.visit_concept_id = ovc.concept_id
  LEFT JOIN `{dataset}.provider` pr
    ON pr.provider_id = vo.provider_id
  LEFT JOIN `{dataset}.concept` sc
    ON sc.concept_id = pr.specialty_concept_id
  GROUP BY b.person_id, b.index_date
),

imaging_utilization AS (
  SELECT
    b.person_id,
    b.index_date,
    COUNTIF(
      po.procedure_date BETWEEN DATE_SUB(b.index_date, INTERVAL @lookback_days DAY) AND b.index_date
      AND (
        LOWER(pc.concept_name) LIKE '%spine%'
        OR LOWER(pc.concept_name) LIKE '%lumbar%'
        OR LOWER(pc.concept_name) LIKE '%cervical%'
        OR LOWER(pc.concept_name) LIKE '%thoracic%'
      )
      AND (
        LOWER(pc.concept_name) LIKE '%mri%'
        OR LOWER(pc.concept_name) LIKE '%ct%'
        OR LOWER(pc.concept_name) LIKE '%x-ray%'
        OR LOWER(pc.concept_name) LIKE '%radiograph%'
      )
    ) AS baseline_spine_imaging,
    COUNTIF(
      po.procedure_date > b.index_date
      AND po.procedure_date <= DATE_ADD(b.index_date, INTERVAL @outcome_window_days DAY)
      AND (
        LOWER(pc.concept_name) LIKE '%spine%'
        OR LOWER(pc.concept_name) LIKE '%lumbar%'
        OR LOWER(pc.concept_name) LIKE '%cervical%'
        OR LOWER(pc.concept_name) LIKE '%thoracic%'
      )
      AND (
        LOWER(pc.concept_name) LIKE '%mri%'
        OR LOWER(pc.concept_name) LIKE '%ct%'
        OR LOWER(pc.concept_name) LIKE '%x-ray%'
        OR LOWER(pc.concept_name) LIKE '%radiograph%'
      )
    ) AS followup_spine_imaging_2y
  FROM baseline_cohort b
  LEFT JOIN `{dataset}.procedure_occurrence` po
    ON po.person_id = b.person_id
  LEFT JOIN `{dataset}.concept` pc
    ON pc.concept_id = po.procedure_concept_id
  GROUP BY b.person_id, b.index_date
),

back_pain_baseline AS (
  SELECT
    b.person_id,
    b.index_date,
    MAX(IF(
      LOWER(c.concept_name) LIKE '%back pain%'
      OR LOWER(c.concept_name) LIKE '%low back pain%'
      OR LOWER(c.concept_name) LIKE '%lumbago%'
      OR LOWER(c.concept_name) LIKE '%sciatica%',
      1, 0
    )) AS baseline_back_pain_flag
  FROM baseline_cohort b
  LEFT JOIN `{dataset}.condition_occurrence` co
    ON co.person_id = b.person_id
   AND co.condition_start_date BETWEEN DATE_SUB(b.index_date, INTERVAL @lookback_days DAY)
                                  AND b.index_date
  LEFT JOIN `{dataset}.concept` c
    ON c.concept_id = co.condition_concept_id
  GROUP BY b.person_id, b.index_date
),

final_cohort_internal AS (
  SELECT
    b.person_id,
    b.exposure_group,
    b.index_date,
    b.observation_period_start_date,
    b.observation_period_end_date,
    DATE_DIFF(b.observation_period_end_date, b.index_date, DAY) AS days_followup,
    IF(DATE_DIFF(b.observation_period_end_date, b.index_date, DAY) >= @outcome_window_days, 1, 0) AS has_min_followup_2y,
    b.birth_date,
    b.age_at_index,
    sex.concept_name AS sex_at_birth,
    race.concept_name AS race,
    eth.concept_name AS ethnicity,
    DATE_DIFF(b.index_date, b.first_t2dm_dx_date, DAY) AS diabetes_duration_days,
    bmi.bmi,
    IF(bmi.bmi IS NOT NULL, 1, 0) AS bmi_present,
    IF(bmi.bmi >= 30, 1, 0) AS obese_bmi30,
    hb.hba1c_recent,
    hb.hba1c_mean_year,
    hb.hba1c_measurements_baseline,
    COALESCE(ins.insulin_use_baseline, 0) AS insulin_use_baseline,
    COALESCE(comp.neuropathy_baseline, 0) AS neuropathy_baseline,
    COALESCE(comp.nephropathy_baseline, 0) AS nephropathy_baseline,
    COALESCE(comp.retinopathy_baseline, 0) AS retinopathy_baseline,
    COALESCE(comp.baseline_condition_count, 0) AS baseline_condition_count,
    COALESCE(util.baseline_outpatient_visits, 0) AS baseline_outpatient_visits,
    COALESCE(util.followup_outpatient_visits_2y, 0) AS followup_outpatient_visits_2y,
    COALESCE(util.baseline_endocrinology_visits, 0) AS baseline_endocrinology_visits,
    COALESCE(util.baseline_orthopedics_visits, 0) AS baseline_orthopedics_visits,
    COALESCE(img.baseline_spine_imaging, 0) AS baseline_spine_imaging,
    COALESCE(img.followup_spine_imaging_2y, 0) AS followup_spine_imaging_2y,
    COALESCE(bp.baseline_back_pain_flag, 0) AS baseline_back_pain_flag,
    sd.first_spine_dx_after_index,
    sd.first_spine_dx_2y,
    IF(sd.first_spine_dx_2y IS NOT NULL, 1, 0) AS incident_spine_2y,
    IF(sd.first_spine_dx_after_index IS NOT NULL, 1, 0) AS event_full_followup,
    IF(
      sd.first_spine_dx_after_index IS NOT NULL,
      DATE_DIFF(sd.first_spine_dx_after_index, b.index_date, DAY),
      DATE_DIFF(b.observation_period_end_date, b.index_date, DAY)
    ) AS time_to_event_or_censor_days,
    LEAST(
      DATE_DIFF(b.observation_period_end_date, b.index_date, DAY),
      @outcome_window_days
    ) AS person_time_2y_days
  FROM baseline_cohort b
  LEFT JOIN `{dataset}.concept` sex ON sex.concept_id = b.sex_at_birth_concept_id
  LEFT JOIN `{dataset}.concept` race ON race.concept_id = b.race_concept_id
  LEFT JOIN `{dataset}.concept` eth ON eth.concept_id = b.ethnicity_concept_id
  LEFT JOIN bmi_baseline bmi
    ON bmi.person_id = b.person_id AND bmi.index_date = b.index_date
  LEFT JOIN hba1c_baseline hb
    ON hb.person_id = b.person_id AND hb.index_date = b.index_date
  LEFT JOIN insulin_baseline ins
    ON ins.person_id = b.person_id AND ins.index_date = b.index_date
  LEFT JOIN complications_baseline comp
    ON comp.person_id = b.person_id AND comp.index_date = b.index_date
  LEFT JOIN utilization util
    ON util.person_id = b.person_id AND util.index_date = b.index_date
  LEFT JOIN imaging_utilization img
    ON img.person_id = b.person_id AND img.index_date = b.index_date
  LEFT JOIN back_pain_baseline bp
    ON bp.person_id = b.person_id AND bp.index_date = b.index_date
  LEFT JOIN spine_dx sd
    ON sd.person_id = b.person_id AND sd.index_date = b.index_date
  LEFT JOIN dm_exclusion_flag dex
    ON dex.person_id = b.person_id
  WHERE dex.person_id IS NULL
    AND sd.spine_dx_on_or_before_index IS NULL
),

final_cohort AS (
  SELECT
    CAST(FARM_FINGERPRINT(CAST(person_id AS STRING)) AS INT64) AS person_id,
    exposure_group,
    index_date,
    observation_period_start_date,
    observation_period_end_date,
    days_followup,
    has_min_followup_2y,
    birth_date,
    age_at_index,
    sex_at_birth,
    race,
    ethnicity,
    diabetes_duration_days,
    bmi,
    bmi_present,
    obese_bmi30,
    hba1c_recent,
    hba1c_mean_year,
    hba1c_measurements_baseline,
    insulin_use_baseline,
    neuropathy_baseline,
    nephropathy_baseline,
    retinopathy_baseline,
    baseline_condition_count,
    baseline_outpatient_visits,
    followup_outpatient_visits_2y,
    baseline_endocrinology_visits,
    baseline_orthopedics_visits,
    baseline_spine_imaging,
    followup_spine_imaging_2y,
    baseline_back_pain_flag,
    first_spine_dx_after_index,
    first_spine_dx_2y,
    incident_spine_2y,
    event_full_followup,
    time_to_event_or_censor_days,
    person_time_2y_days
  FROM final_cohort_internal
)
"""


def build_analytic_dataset_sql(dataset: str) -> str:
    return _core_ctes(dataset) + "\nSELECT * FROM final_cohort"


def build_cohort_flow_sql(dataset: str) -> str:
    return (
        _core_ctes(dataset)
        + """
SELECT '01_drug_start_any' AS step, COUNT(DISTINCT person_id) AS n
FROM drug_starts
WHERE metformin_start_date IS NOT NULL OR glp1_start_date IS NOT NULL
UNION ALL
SELECT '02_exposure_group_assigned', COUNT(DISTINCT person_id)
FROM exposure_clean
UNION ALL
SELECT '03_t2dm_prior_to_index', COUNT(DISTINCT person_id)
FROM t2dm_eligible
UNION ALL
SELECT '04_adult_population', COUNT(DISTINCT person_id)
FROM adult_population
UNION ALL
SELECT '05_observation_lookback_eligible', COUNT(DISTINCT person_id)
FROM baseline_cohort
UNION ALL
SELECT '06_after_dm_exclusions', COUNT(DISTINCT b.person_id)
FROM baseline_cohort b
LEFT JOIN dm_exclusion_flag d ON d.person_id = b.person_id
WHERE d.person_id IS NULL
UNION ALL
SELECT '07_after_prevalent_spine_exclusion', COUNT(DISTINCT b.person_id)
FROM baseline_cohort b
LEFT JOIN spine_dx s
  ON s.person_id = b.person_id AND s.index_date = b.index_date
LEFT JOIN dm_exclusion_flag d
  ON d.person_id = b.person_id
WHERE d.person_id IS NULL
  AND s.spine_dx_on_or_before_index IS NULL
UNION ALL
SELECT '08_final_analytic_cohort', COUNT(DISTINCT person_id)
FROM final_cohort
UNION ALL
SELECT '09_has_730d_followup', COUNT(DISTINCT person_id)
FROM final_cohort
WHERE has_min_followup_2y = 1
"""
    )


def build_post_index_bmi_sql(dataset: str) -> str:
    return (
        _core_ctes(dataset)
        + """
SELECT
  CAST(FARM_FINGERPRINT(CAST(fc.person_id AS STRING)) AS INT64) AS person_id,
  fc.exposure_group,
  fc.index_date,
  m.measurement_date,
  m.value_as_number AS bmi
FROM final_cohort_internal fc
JOIN `{dataset}.measurement` m
  ON m.person_id = fc.person_id
 AND m.measurement_concept_id IN UNNEST(@bmi_measurement_concept_ids)
 AND m.value_as_number BETWEEN @bmi_min AND @bmi_max
 AND m.measurement_date > fc.index_date
 AND m.measurement_date <= DATE_ADD(fc.index_date, INTERVAL @outcome_window_days DAY)
""".format(dataset=dataset)
    )


def fetch_cohort_data(client: bigquery.Client, dataset: str, config: dict, concepts: dict) -> CohortData:
    params = _query_params(config, concepts)
    flow_sql = build_cohort_flow_sql(dataset)
    analytic_sql = build_analytic_dataset_sql(dataset)
    post_index_bmi_sql = build_post_index_bmi_sql(dataset)

    cohort_flow = run_query(
        client,
        flow_sql,
        params,
        job_name="project6_cohort_flow",
    )

    temp_dataset = str(config.get("temp_dataset", "")).strip()
    if temp_dataset:
        analytic_table = f"{temp_dataset}.project6_analytic_cache"
        bmi_table = f"{temp_dataset}.project6_post_index_bmi_cache"
        materialize_query(
            client,
            analytic_sql,
            params,
            table_fqn=analytic_table,
            job_name="project6_materialize_analytic_cache",
        )
        materialize_query(
            client,
            post_index_bmi_sql,
            params,
            table_fqn=bmi_table,
            job_name="project6_materialize_post_index_bmi_cache",
        )
        analytic_df = run_query(
            client,
            f"SELECT * FROM `{analytic_table}`",
            None,
            job_name="project6_read_analytic_cache",
        )
        post_index_bmi_df = run_query(
            client,
            f"SELECT * FROM `{bmi_table}`",
            None,
            job_name="project6_read_post_index_bmi_cache",
        )
    else:
        analytic_df = run_query(
            client,
            analytic_sql,
            params,
            job_name="project6_analytic_dataset",
        )
        post_index_bmi_df = run_query(
            client,
            post_index_bmi_sql,
            params,
            job_name="project6_post_index_bmi",
        )

    date_cols = [
        "index_date",
        "observation_period_start_date",
        "observation_period_end_date",
        "birth_date",
        "first_spine_dx_after_index",
        "first_spine_dx_2y",
    ]
    for col in date_cols:
        if col in analytic_df.columns:
            analytic_df[col] = pd.to_datetime(analytic_df[col], errors="coerce")

    if "measurement_date" in post_index_bmi_df.columns:
        post_index_bmi_df["measurement_date"] = pd.to_datetime(
            post_index_bmi_df["measurement_date"], errors="coerce"
        )
    if "index_date" in post_index_bmi_df.columns:
        post_index_bmi_df["index_date"] = pd.to_datetime(post_index_bmi_df["index_date"], errors="coerce")

    return CohortData(
        cohort_flow=cohort_flow,
        analytic_df=analytic_df,
        post_index_bmi_df=post_index_bmi_df,
    )
