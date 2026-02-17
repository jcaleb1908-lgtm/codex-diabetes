"""Configuration for Project 6 degenerative spine + diabetes medication analyses."""

from __future__ import annotations

import os
from pathlib import Path

CHANGE_LOG = [
    "2026-02-17: Second-pass tuning set overlap-first weighting defaults with ESS-floor candidate selection and tighter truncation options.",
    "2026-02-17: Rebuilt PROJECT6_SPINE_ALL_IN_ONE.py as a true standalone monolith generated from the modular pipeline.",
    "2026-02-17: Added MI-first handling for BMI/HbA1c missingness with pooled PS/outcome estimates plus double-weighting sensitivity.",
    "2026-02-17: Added overlap/ATT/IPTW weighting selection by best post-weight balance and explicit extreme-weight diagnostics (max/p95/p99/ESS).",
    "2026-02-17: Added HbA1c unit/range normalization, winsorization guards, and Cox penalization defaults for sparse-event stability.",
    "2026-02-16: Refactored SPINE.rtf notebook-style code into a modular pipeline with a single main() entrypoint.",
    "2026-02-16: Preserved prior treatment group logic (metformin_only, glp1_only, combo) and made combo window configurable.",
    "2026-02-16: Preserved prior T2DM anchor concept (201826) and prior diabetes exclusion anchors (24612, 313217, 199074).",
    "2026-02-16: Preserved prior spine concept set IDs from existing code and exposed them in config for transparent versioning.",
    "2026-02-16: Added explicit prevalence exclusion at/before index and incident definition > index with 730-day binary window.",
    "2026-02-16: Added centralized All of Us n<20 suppression and model-level subgroup/event guardrails.",
    "2026-02-16: Added BMI missingness diagnostics, BMI-observed IPW, and pattern-mixture sensitivity analyses.",
    "2026-02-16: Added confounding-by-indication severity proxies + propensity-score IPTW + balance diagnostics.",
    "2026-02-16: Added surveillance/utilization bias analyses and utilization-stratified sensitivity models.",
    "2026-02-16: Added Cox time-to-event models with 2-year censoring plus logistic 2-year replication models.",
    "2026-02-16: Added post-index weight-change extraction (6/12/24 months) and exploratory outcome analyses.",
]

ASSUMPTIONS = [
    "Execution is inside All of Us Researcher Workbench Controlled Tier with BigQuery credentials and WORKSPACE_CDR set.",
    "Medication exposure is assigned from earliest qualifying drug_exposure_start_date for metformin and GLP-1 descendants.",
    "Combo therapy is defined by metformin and GLP-1 starts within combo_window_days of each other.",
    "T2DM case definition uses condition-occurrence descendants of anchor concept 201826 before/on index date.",
    "Diabetes exclusions use descendants of concept anchors retained from existing code (24612, 313217, 199074).",
    "Spine outcome concept list currently mirrors the existing code export; update config when final 13-SNOMED set is finalized.",
    "Cox models use 2-year censoring and include all participants with >0 observed follow-up; binary 2-year models can remain restricted.",
    "PH assumption diagnostics prefer lifelines when available; otherwise fallback checks are limited and reported.",
    "Primary causal models use multiple imputation for BMI/HbA1c plus overlap/robust weighting; double-weighting for missingness is retained as sensitivity.",
]

CONCEPTS = {
    "T2DM_ANCESTOR_CONCEPTS": [201826],
    "DM_EXCLUSION_ANCESTOR_CONCEPTS": [24612, 313217, 199074],
    "METFORMIN_INGREDIENT_NAMES": ["metformin"],
    "GLP1_INGREDIENT_NAMES": [
        "liraglutide",
        "semaglutide",
        "dulaglutide",
        "exenatide",
        "lixisenatide",
        "albiglutide",
        "tirzepatide",
    ],
    # Preserved from prior SPINE code export.
    "SPINE_OUTCOME_SNOMED_CONCEPT_IDS": [
        137548,
        198520,
        36717608,
        4134121,
        4134122,
        4167097,
        4187244,
        608177,
        761918,
        77079,
        79119,
        80816,
    ],
    "BMI_MEASUREMENT_CONCEPT_IDS": [3038553, 4245997, 40762636],
    "HBA1C_MEASUREMENT_CONCEPT_IDS": [3004410],
    "OUTPATIENT_VISIT_ANCESTOR_CONCEPT_IDS": [9202],
}

CONFIG = {
    "dataset": os.environ.get("WORKSPACE_CDR", "").strip(),
    "bq_location": os.environ.get("GOOGLE_CLOUD_REGION", "US"),
    # Optional writable dataset for cached intermediate tables.
    "temp_dataset": os.environ.get("WORKSPACE_TEMP_DATASET", "").strip(),
    "random_seed": 42,
    "adult_age_min": 18,
    "adult_age_max": 90,
    "lookback_days": 365,
    "combo_window_days": 30,
    "outcome_window_days": 730,
    "bmi_baseline_window_start_days": -365,
    "bmi_baseline_window_end_days": 30,
    "bmi_min": 10.0,
    "bmi_max": 80.0,
    "bmi_winsor_quantiles": (0.005, 0.995),
    "bmi_center_value": 30.0,
    "hba1c_plausible_min": 3.0,
    "hba1c_plausible_max": 20.0,
    "hba1c_winsor_quantiles": (0.005, 0.995),
    "duration_winsor_quantiles": (0.005, 0.995),
    "utilization_winsor_quantiles": (0.005, 0.995),
    "require_730d_for_binary_models": True,
    "cox_time_horizon_days": 730,
    "cox_require_positive_followup_days": True,
    "cox_penalizer": 0.1,
    "logit_event_cell_warn_threshold": 5,
    "logit_events_per_parameter_warn_threshold": 10.0,
    "logit_penalty_alpha": 1e-3,
    "logit_penalty_maxiter": 2000,
    "force_penalized_logit_models": [
        "logistic_interaction_obesity",
        "logistic_interaction_continuous_bmi",
        "utilization_adjusted_logit",
    ],
    "force_penalized_logit_prefixes": [
        "stratified_logit_",
        "utilization_tertile_",
    ],
    "print_tables_in_notebook": True,
    "print_table_max_rows": 30,
    "reference_levels": {
        "exposure_group": "metformin_only",
        "sex_simple": "Male",
        "race_simple": "White",
        "ethnicity_simple": "Hispanic or Latino",
    },
    "category_levels": {
        "exposure_group": ["metformin_only", "glp1_only", "combo"],
        "sex_simple": ["Male", "Female", "Other/Unknown"],
        "race_simple": ["White", "Black or African American", "Asian", "Other/Unknown"],
        "ethnicity_simple": ["Hispanic or Latino", "Not Hispanic or Latino", "Unknown"],
    },
    "required_analysis_columns": [
        "person_id",
        "exposure_group",
        "days_followup",
        "incident_spine_2y",
        "event_full_followup",
        "time_to_event_or_censor_days",
        "person_time_2y_days",
        "has_min_followup_2y",
        "age_at_index",
        "bmi",
    ],
    "small_cell_threshold": 20,
    "mi_num_imputations": 5,
    "mi_max_iter": 20,
    "mi_target_columns": ["bmi", "hba1c_recent", "hba1c_mean_year"],
    "missingness_weight_truncation_quantiles": (0.01, 0.99),
    "ps_weighting_strategies": ["overlap"],
    "ps_stabilized_weights": True,
    "ps_weight_truncation_options": [(0.01, 0.99), (0.005, 0.995), (0.02, 0.98)],
    "ps_weight_truncation_quantiles": (0.01, 0.99),
    "ps_balance_target_abs_smd": 0.10,
    "ps_min_ess_ratio": 0.30,
    "ps_use_ml_if_available": True,
    "ps_ml_n_estimators": 300,
    "ps_ml_learning_rate": 0.05,
    "ps_ml_subsample": 0.8,
    "weight_change_target_days": [180, 365, 730],
    "weight_change_window_tolerance_days": 60,
    "output_dir": str(Path(__file__).resolve().parents[1] / "project6_outputs"),
}

REQUIRED_OUTPUT_FILES = [
    "cohort_flow.csv",
    "table1_baseline_by_treatment.csv",
    "severity_baseline_table.csv",
    "bmi_missingness_table.csv",
    "complete_vs_missing_bmi_risk.csv",
    "logistic_main_hc3.csv",
    "logistic_interaction_obesity.csv",
    "interaction_results.csv",
    "logit_model_diagnostics.csv",
    "cox_model_diagnostics.csv",
    "cox_time_to_spine.csv",
    "propensity_score_results.csv",
    "balance_diagnostics.csv",
    "utilization_bias_results.csv",
    "weight_change_analysis.csv",
    "forest_plot_ready.csv",
    "km_curve_data.csv",
    "REPORT.md",
]


def validate_config() -> None:
    dataset = CONFIG["dataset"]
    if not dataset:
        raise ValueError("WORKSPACE_CDR is empty. Set WORKSPACE_CDR before running.")


def ensure_output_dir() -> Path:
    out_dir = Path(CONFIG["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
