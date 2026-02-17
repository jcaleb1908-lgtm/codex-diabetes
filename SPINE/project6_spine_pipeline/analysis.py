"""Analysis modules for Project 6 spine outcomes."""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError, PerfectSeparationWarning

from .suppression import POLICY_NOTE, model_is_policy_compliant

try:
    from scipy.stats import chi2_contingency
except Exception:  # pragma: no cover - optional dependency
    chi2_contingency = None

try:
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
except Exception:  # pragma: no cover - optional dependency
    CoxPHFitter = None
    proportional_hazard_test = None

try:
    from statsmodels.duration.hazard_regression import PHReg
except Exception:  # pragma: no cover - optional dependency
    PHReg = None

try:  # pragma: no cover - optional dependency
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
except Exception:  # pragma: no cover - optional dependency
    IterativeImputer = None

try:  # pragma: no cover - optional dependency
    from sklearn.ensemble import GradientBoostingClassifier
except Exception:  # pragma: no cover - optional dependency
    GradientBoostingClassifier = None


@dataclass
class AnalysisBundle:
    table1: pd.DataFrame
    severity_table: pd.DataFrame
    bmi_missingness: pd.DataFrame
    complete_vs_missing: pd.DataFrame
    logistic_main: pd.DataFrame
    logistic_interaction: pd.DataFrame
    interaction_results: pd.DataFrame
    cox_results: pd.DataFrame
    ps_results: pd.DataFrame
    balance_diagnostics: pd.DataFrame
    utilization_results: pd.DataFrame
    weight_change_results: pd.DataFrame
    forest_ready: pd.DataFrame
    km_curve_data: pd.DataFrame
    suppression_exclusions: pd.DataFrame
    logit_diagnostics: pd.DataFrame
    cox_diagnostics: pd.DataFrame
    notes: list[str]
    artifacts: dict[str, object]


def _safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _winsorize_series(series: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype(float).copy()
    non_null = out.dropna()
    if non_null.empty:
        return out
    lo = float(non_null.quantile(lower_q))
    hi = float(non_null.quantile(upper_q))
    return out.clip(lower=lo, upper=hi)


def _normalize_hba1c_series(series: pd.Series, config: dict, notes: list[str] | None = None, *, label: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").copy()
    raw_non_null = int(out.notna().sum())
    if raw_non_null == 0:
        return out

    # Heuristic rescaling for mixed encodings seen in AoU exports.
    scale_10 = out.gt(20) & out.le(200)
    scale_100 = out.gt(200) & out.le(2000)
    scale_1000 = out.gt(2000) & out.le(200000)
    out.loc[scale_10] = out.loc[scale_10] / 10.0
    out.loc[scale_100] = out.loc[scale_100] / 100.0
    out.loc[scale_1000] = out.loc[scale_1000] / 1000.0

    plausible_min = float(config.get("hba1c_plausible_min", 3.0))
    plausible_max = float(config.get("hba1c_plausible_max", 20.0))
    invalid_mask = out.notna() & ~out.between(plausible_min, plausible_max)
    dropped = int(invalid_mask.sum())
    out.loc[invalid_mask] = np.nan

    low_q, high_q = config.get("hba1c_winsor_quantiles", (0.005, 0.995))
    out = _winsorize_series(out, float(low_q), float(high_q))

    scaled_n = int(scale_10.sum() + scale_100.sum() + scale_1000.sum())
    if notes is not None and (scaled_n > 0 or dropped > 0):
        notes.append(
            f"{label}: normalized HBA1c scaling for {scaled_n} rows and dropped {dropped} implausible rows."
        )
    return out


def _two_sided_p_from_z(z_value: float) -> float:
    if not np.isfinite(z_value):
        return np.nan
    return float(math.erfc(abs(float(z_value)) / math.sqrt(2.0)))


def _effective_sample_size(weights: pd.Series) -> float:
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    if w.empty:
        return 0.0
    denom = float(np.sum(np.square(w)))
    if denom <= 0:
        return 0.0
    num = float(np.sum(w))
    return float((num * num) / denom)


def _category_levels(config: dict, field: str, fallback: list[str]) -> list[str]:
    raw_levels = config.get("category_levels", {}).get(field)
    if raw_levels:
        return [str(x) for x in raw_levels]
    return fallback


def _reference_term(field: str, config: dict) -> str:
    ref = str(config.get("reference_levels", {}).get(field, "")).strip()
    if ref:
        return f'C({field}, Treatment(reference="{ref}"))'
    return f"C({field})"


def _base_logit_terms(config: dict, *, include_exposure: bool = True) -> list[str]:
    terms: list[str] = []
    if include_exposure:
        terms.append(_reference_term("exposure_group", config))
    terms.extend(
        [
            _reference_term("sex_simple", config),
            _reference_term("race_simple", config),
            _reference_term("ethnicity_simple", config),
            "age_at_index",
        ]
    )
    return terms


def _ensure_required_columns(df: pd.DataFrame, config: dict, notes: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    required_cols = [str(c) for c in config.get("required_analysis_columns", [])]
    if not required_cols:
        return out

    missing = [col for col in required_cols if col not in out.columns]
    if not missing:
        return out

    for col in missing:
        out[col] = np.nan
    msg = f"prepare_analysis_df: required columns missing and filled as NaN: {', '.join(sorted(missing))}"
    logging.warning(msg)
    if notes is not None:
        notes.append(msg)
    return out


def _standardize_sex(x: object) -> str:
    if pd.isna(x):
        return "Other/Unknown"
    s = str(x).strip().lower()
    if s == "male":
        return "Male"
    if s == "female":
        return "Female"
    return "Other/Unknown"


def _standardize_race(x: object) -> str:
    if pd.isna(x):
        return "Other/Unknown"
    s = str(x).strip().lower()
    if "white" in s:
        return "White"
    if "black" in s or "african" in s:
        return "Black or African American"
    if "asian" in s:
        return "Asian"
    return "Other/Unknown"


def _standardize_ethnicity(x: object) -> str:
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip().lower()
    if "not hispanic" in s:
        return "Not Hispanic or Latino"
    if "hispanic" in s or "latino" in s:
        return "Hispanic or Latino"
    return "Unknown"


def _age_bin(x: float) -> str:
    if pd.isna(x):
        return "Unknown"
    if x < 40:
        return "18-39"
    if x < 50:
        return "40-49"
    if x < 60:
        return "50-59"
    if x < 70:
        return "60-69"
    return "70+"


def prepare_analysis_df(df: pd.DataFrame, config: dict, notes: list[str] | None = None) -> pd.DataFrame:
    out = _ensure_required_columns(df, config, notes=notes)
    out = _safe_numeric(
        out,
        [
            "age_at_index",
            "bmi",
            "days_followup",
            "incident_spine_2y",
            "event_full_followup",
            "time_to_event_or_censor_days",
            "person_time_2y_days",
            "hba1c_recent",
            "hba1c_mean_year",
            "diabetes_duration_days",
            "insulin_use_baseline",
            "neuropathy_baseline",
            "nephropathy_baseline",
            "retinopathy_baseline",
            "baseline_condition_count",
            "baseline_outpatient_visits",
            "followup_outpatient_visits_2y",
            "baseline_spine_imaging",
            "followup_spine_imaging_2y",
            "baseline_endocrinology_visits",
            "baseline_orthopedics_visits",
            "baseline_back_pain_flag",
            "has_min_followup_2y",
        ],
    )

    for col in ["index_date", "first_spine_dx_after_index", "first_spine_dx_2y"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    out["incident_spine_2y"] = out["incident_spine_2y"].fillna(0).astype(int)
    out["event_full_followup"] = out["event_full_followup"].fillna(0).astype(int)
    out["has_min_followup_2y"] = out["has_min_followup_2y"].fillna(0).astype(int)

    if "exposure_group" not in out.columns:
        out["exposure_group"] = np.nan

    exposure_levels = _category_levels(config, "exposure_group", ["metformin_only", "glp1_only", "combo"])
    exposure_raw = out["exposure_group"].astype(str).str.strip().str.lower()
    exposure_missing = out["exposure_group"].isna()
    unexpected_mask = (~exposure_missing) & (~exposure_raw.isin(exposure_levels))
    unexpected_count = int(unexpected_mask.sum())
    if unexpected_count:
        bad_values = sorted(set(exposure_raw.loc[unexpected_mask].tolist()))
        preview = ", ".join(bad_values[:5])
        msg = (
            f"Found {unexpected_count} rows with unexpected exposure_group values "
            f"({preview}); setting them to missing for analysis."
        )
        logging.warning(msg)
        if notes is not None:
            notes.append(msg)
        out.loc[unexpected_mask, "exposure_group"] = np.nan

    out.loc[out["exposure_group"].notna(), "exposure_group"] = (
        out.loc[out["exposure_group"].notna(), "exposure_group"].astype(str).str.strip().str.lower()
    )

    sex_src = out["sex_at_birth"] if "sex_at_birth" in out.columns else pd.Series(np.nan, index=out.index)
    race_src = out["race"] if "race" in out.columns else pd.Series(np.nan, index=out.index)
    eth_src = out["ethnicity"] if "ethnicity" in out.columns else pd.Series(np.nan, index=out.index)

    out["sex_simple"] = sex_src.map(_standardize_sex)
    out["race_simple"] = race_src.map(_standardize_race)
    out["ethnicity_simple"] = eth_src.map(_standardize_ethnicity)

    for hcol in ["hba1c_recent", "hba1c_mean_year"]:
        if hcol in out.columns:
            out[hcol] = _normalize_hba1c_series(out[hcol], config, notes=notes, label=hcol)

    out["bmi_present"] = out["bmi"].notna().astype(int)
    out["bmi_missing"] = (out["bmi_present"] == 0).astype(int)
    out["obese_bmi30"] = np.where(out["bmi"].notna() & (out["bmi"] >= 30), 1, 0)

    out["age_bin"] = out["age_at_index"].apply(_age_bin)
    base_outpatient = out["baseline_outpatient_visits"] if "baseline_outpatient_visits" in out.columns else 0
    base_spine_imaging = out["baseline_spine_imaging"] if "baseline_spine_imaging" in out.columns else 0
    base_endo = out["baseline_endocrinology_visits"] if "baseline_endocrinology_visits" in out.columns else 0
    base_ortho = out["baseline_orthopedics_visits"] if "baseline_orthopedics_visits" in out.columns else 0
    out["utilization_total_baseline"] = (
        pd.Series(base_outpatient, index=out.index).fillna(0)
        + pd.Series(base_spine_imaging, index=out.index).fillna(0)
        + pd.Series(base_endo, index=out.index).fillna(0)
        + pd.Series(base_ortho, index=out.index).fillna(0)
    )
    util_winsor = config.get("utilization_winsor_quantiles", (0.005, 0.995))
    for ucol in [
        "baseline_outpatient_visits",
        "followup_outpatient_visits_2y",
        "baseline_spine_imaging",
        "followup_spine_imaging_2y",
        "baseline_endocrinology_visits",
        "baseline_orthopedics_visits",
        "utilization_total_baseline",
    ]:
        if ucol in out.columns:
            out[ucol] = out[ucol].clip(lower=0)
            out[ucol] = _winsorize_series(out[ucol], float(util_winsor[0]), float(util_winsor[1]))

    if "diabetes_duration_days" in out.columns:
        out.loc[out["diabetes_duration_days"] < 0, "diabetes_duration_days"] = np.nan
        dur_winsor = config.get("duration_winsor_quantiles", (0.005, 0.995))
        out["diabetes_duration_days"] = _winsorize_series(
            out["diabetes_duration_days"], float(dur_winsor[0]), float(dur_winsor[1])
        )
        out["diabetes_duration_years"] = out["diabetes_duration_days"] / 365.25
    else:
        out["diabetes_duration_years"] = np.nan

    # Keep implausible BMI out of analyses while preserving BMI-missing diagnostics.
    lo, hi = float(config["bmi_min"]), float(config["bmi_max"])
    out.loc[(out["bmi"].notna()) & ((out["bmi"] < lo) | (out["bmi"] > hi)), "bmi"] = np.nan
    out["bmi_present"] = out["bmi"].notna().astype(int)
    out["bmi_missing"] = 1 - out["bmi_present"]
    out["obese_bmi30"] = np.where(out["bmi"].notna() & (out["bmi"] >= 30), 1, 0)
    bmi_center = float(config.get("bmi_center_value", 30.0))
    out["bmi_c"] = out["bmi"] - bmi_center
    bmi_winsor = config.get("bmi_winsor_quantiles", (0.005, 0.995))
    out["bmi"] = _winsorize_series(out["bmi"], float(bmi_winsor[0]), float(bmi_winsor[1]))
    out["bmi_c"] = out["bmi"] - bmi_center

    # Collapse sparse microvascular flags to stabilize sparse-outcome models.
    micro_cols = [c for c in ["neuropathy_baseline", "nephropathy_baseline", "retinopathy_baseline"] if c in out.columns]
    for col in micro_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).clip(0, 1)
    if micro_cols:
        out["microvascular_any_baseline"] = (out[micro_cols].sum(axis=1) > 0).astype(int)
    else:
        out["microvascular_any_baseline"] = 0

    out["sex_female_flag"] = (out["sex_simple"] == "Female").astype(int)
    out["age_sq"] = out["age_at_index"] ** 2
    out["bmi_sq"] = out["bmi"] ** 2
    out["duration_sq"] = out["diabetes_duration_days"] ** 2
    outpatient = (
        pd.to_numeric(out["baseline_outpatient_visits"], errors="coerce")
        if "baseline_outpatient_visits" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    insulin = (
        pd.to_numeric(out["insulin_use_baseline"], errors="coerce")
        if "insulin_use_baseline" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    comorbidity = (
        pd.to_numeric(out["baseline_condition_count"], errors="coerce")
        if "baseline_condition_count" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    out["outpatient_visits_sq"] = outpatient ** 2
    out["duration_x_insulin"] = out["diabetes_duration_days"] * insulin.fillna(0)
    out["age_x_comorbidity"] = out["age_at_index"] * comorbidity.fillna(0)
    out["sex_female_x_bmi"] = out["sex_female_flag"] * out["bmi"]

    # Stable categories for reference groups.
    out["exposure_group"] = pd.Categorical(
        out["exposure_group"],
        categories=exposure_levels,
    )
    out["sex_simple"] = pd.Categorical(
        out["sex_simple"],
        categories=_category_levels(config, "sex_simple", ["Male", "Female", "Other/Unknown"]),
    )
    out["race_simple"] = pd.Categorical(
        out["race_simple"],
        categories=_category_levels(
            config,
            "race_simple",
            ["White", "Black or African American", "Asian", "Other/Unknown"],
        ),
    )
    out["ethnicity_simple"] = pd.Categorical(
        out["ethnicity_simple"],
        categories=_category_levels(
            config,
            "ethnicity_simple",
            ["Hispanic or Latino", "Not Hispanic or Latino", "Unknown"],
        ),
    )

    return out


def _logit_or_table(fit, label: str) -> pd.DataFrame:
    conf = fit.conf_int()
    stderr = fit.bse.values if hasattr(fit, "bse") else np.repeat(np.nan, len(fit.params))
    out = pd.DataFrame(
        {
            "term": fit.params.index,
            "coef": fit.params.values,
            "std_error": stderr,
            "or": np.exp(fit.params.values),
            "ci_low": np.exp(conf[0].values),
            "ci_high": np.exp(conf[1].values),
            "p_value": fit.pvalues.values,
            "model": label,
            "effect_type": "OR",
            "effect_scale": "log",
        }
    )
    return out


def _glm_or_table(fit, label: str) -> pd.DataFrame:
    params = fit.params
    conf = fit.conf_int()
    stderr = fit.bse.values if hasattr(fit, "bse") else np.repeat(np.nan, len(params))
    out = pd.DataFrame(
        {
            "term": params.index,
            "coef": params.values,
            "std_error": stderr,
            "or": np.exp(params.values),
            "ci_low": np.exp(conf[0].values),
            "ci_high": np.exp(conf[1].values),
            "p_value": fit.pvalues.values,
            "model": label,
            "effect_type": "OR",
            "effect_scale": "log",
        }
    )
    return out


def _create_policy_note_df(name: str) -> pd.DataFrame:
    return pd.DataFrame({"analysis": [name], "policy_note": [POLICY_NOTE]})


def _fit_regularized_logit(model, label: str, alpha: float, maxiter: int) -> tuple[pd.DataFrame, object]:
    # Suppress known solver chatter; hard failures still raise exceptions.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        fit = model.fit_regularized(alpha=alpha, L1_wt=0.0, disp=False, maxiter=maxiter)

    names = getattr(model, "exog_names", None)
    raw_params = np.asarray(fit.params).reshape(-1)
    if names is None or len(names) != len(raw_params):
        names = [f"param_{i}" for i in range(len(raw_params))]
    params = pd.Series(raw_params, index=names)

    out = pd.DataFrame(
        {
            "term": params.index,
            "coef": params.values,
            "std_error": np.nan,
            "or": np.exp(params.values),
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "model": label,
            "estimator": "penalized_logit_ridge",
            "penalty_alpha": alpha,
            "effect_type": "OR",
            "effect_scale": "log",
        }
    )
    return out, fit


def _should_force_penalized(label: str, config: dict | None) -> bool:
    cfg = config or {}
    forced_labels = {str(x) for x in cfg.get("force_penalized_logit_models", [])}
    if label in forced_labels:
        return True
    forced_prefixes = [str(x) for x in cfg.get("force_penalized_logit_prefixes", [])]
    return any(label.startswith(prefix) for prefix in forced_prefixes)


def _penalized_logit_with_note(
    *,
    model,
    label: str,
    reason: str,
    config: dict | None,
    notes: list[str],
    model_store: dict[str, object] | None,
) -> pd.DataFrame:
    alpha = float((config or {}).get("logit_penalty_alpha", 1e-3))
    maxiter = int((config or {}).get("logit_penalty_maxiter", 2000))
    notes.append(
        f"{label}: {reason}; using ridge-penalized logit (alpha={alpha}, maxiter={maxiter})."
    )
    try:
        out, fit = _fit_regularized_logit(model, label=label, alpha=alpha, maxiter=maxiter)
        if model_store is not None:
            model_store[label] = fit
        return out
    except Exception as penalized_exc:  # pragma: no cover - numerical edge cases
        notes.append(f"{label}: penalized fallback failed ({penalized_exc}).")
        return pd.DataFrame({"analysis": [label], "error": [str(penalized_exc)]})


def _build_logit_diagnostics(
    data: pd.DataFrame,
    *,
    label: str,
    event_col: str,
    parameter_count: int | None,
    config: dict | None = None,
) -> pd.DataFrame:
    tiny_threshold = int((config or {}).get("logit_event_cell_warn_threshold", 5))
    epv_warn = float((config or {}).get("logit_events_per_parameter_warn_threshold", 10.0))

    n = int(len(data))
    events = int(data[event_col].sum()) if event_col in data.columns else 0
    event_rate = (events / n) if n else np.nan

    diag_rows: list[dict[str, object]] = [
        {
            "analysis": label,
            "scope": "overall",
            "level": "overall",
            "n": n,
            "events": events,
            "nonevents": int(n - events),
            "event_rate": event_rate,
            "n_parameters": parameter_count,
        }
    ]
    logging.info("%s: n=%s events=%s event_rate=%.4f", label, n, events, event_rate if n else 0.0)

    if "exposure_group" in data.columns:
        by_exp = (
            data.groupby("exposure_group", dropna=False, observed=False)
            .agg(n=(event_col, "size"), events=(event_col, "sum"))
            .reset_index()
        )
        if not by_exp.empty:
            by_exp["nonevents"] = by_exp["n"] - by_exp["events"]
            by_exp["event_rate"] = by_exp["events"] / by_exp["n"]
            for _, row in by_exp.iterrows():
                diag_rows.append(
                    {
                        "analysis": label,
                        "scope": "by_exposure",
                        "level": str(row["exposure_group"]),
                        "n": int(row["n"]),
                        "events": int(row["events"]),
                        "nonevents": int(row["nonevents"]),
                        "event_rate": float(row["event_rate"]) if pd.notna(row["event_rate"]) else np.nan,
                        "n_parameters": parameter_count,
                    }
                )
            tiny = by_exp.loc[(by_exp["events"] < tiny_threshold) | (by_exp["nonevents"] < tiny_threshold)]
            if not tiny.empty:
                tiny_levels = ", ".join(str(x) for x in tiny["exposure_group"].tolist())
                logging.warning(
                    "%s: tiny event/non-event cells by exposure (<%s): %s",
                    label,
                    tiny_threshold,
                    tiny_levels,
                )

    if parameter_count is not None:
        denom = max(parameter_count - 1, 1)
        epv = min(events, n - events) / denom if n else np.nan
        logging.info("%s: parameters=%s EPV=%.3f", label, parameter_count, epv if pd.notna(epv) else np.nan)
        if pd.notna(epv) and epv < epv_warn:
            logging.warning("%s: low events-per-parameter (%.3f < %.3f)", label, epv, epv_warn)

    return pd.DataFrame(diag_rows)


def _fit_logit_if_allowed(
    data: pd.DataFrame,
    formula: str,
    *,
    label: str,
    event_col: str,
    threshold: int,
    notes: list[str],
    config: dict | None = None,
    model_store: dict[str, object] | None = None,
    diagnostics_store: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    n = int(len(data))
    events = int(data[event_col].sum()) if event_col in data.columns else 0

    try:
        model = smf.logit(formula=formula, data=data)
        n_params = int(model.exog.shape[1]) if hasattr(model, "exog") else None
    except Exception as exc:  # pragma: no cover
        notes.append(f"{label}: could not build design matrix ({exc}).")
        return pd.DataFrame({"analysis": [label], "error": [f"design matrix build failed: {exc}"]})

    diag = _build_logit_diagnostics(
        data=data,
        label=label,
        event_col=event_col,
        parameter_count=n_params,
        config=config,
    )
    if diagnostics_store is not None:
        diagnostics_store.append(diag)

    compliant, note = model_is_policy_compliant(n=n, events=events, threshold=threshold)
    if not compliant:
        notes.append(f"{label}: {note}")
        return _create_policy_note_df(label)

    if _should_force_penalized(label, config):
        return _penalized_logit_with_note(
            model=model,
            label=label,
            reason="configured to use penalized estimation",
            config=config,
            notes=notes,
            model_store=model_store,
        )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=PerfectSeparationWarning)
            warnings.filterwarnings("error", category=ConvergenceWarning)
            warnings.filterwarnings("error", category=RuntimeWarning)
            fit = model.fit(disp=False, cov_type="HC3")
        mle_retvals = getattr(fit, "mle_retvals", {})
        if isinstance(mle_retvals, dict) and not bool(mle_retvals.get("converged", True)):
            return _penalized_logit_with_note(
                model=model,
                label=label,
                reason="standard MLE did not converge",
                config=config,
                notes=notes,
                model_store=model_store,
            )
        if model_store is not None:
            model_store[label] = fit
        return _logit_or_table(fit, label)
    except (
        PerfectSeparationError,
        PerfectSeparationWarning,
        ConvergenceWarning,
        np.linalg.LinAlgError,
        RuntimeWarning,
        OverflowError,
    ) as exc:  # pragma: no cover - depends on data
        return _penalized_logit_with_note(
            model=model,
            label=label,
            reason=f"unstable standard MLE ({exc})",
            config=config,
            notes=notes,
            model_store=model_store,
        )
    except Exception as exc:  # pragma: no cover - numerical edge cases
        notes.append(f"{label}: model failed ({exc}).")
        return pd.DataFrame({"analysis": [label], "error": [str(exc)]})


def _risk_table_by_group(df: pd.DataFrame, group_col: str, event_col: str) -> pd.DataFrame:
    out = (
        df.groupby(group_col, dropna=False)
        .agg(n=(event_col, "size"), events=(event_col, "sum"))
        .reset_index()
    )
    out["risk"] = out["events"] / out["n"]
    return out


def build_table1(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    numeric_vars = [
        "age_at_index",
        "bmi",
        "hba1c_recent",
        "hba1c_mean_year",
        "diabetes_duration_days",
        "baseline_condition_count",
        "baseline_outpatient_visits",
        "baseline_spine_imaging",
        "baseline_endocrinology_visits",
        "baseline_orthopedics_visits",
    ]
    cat_vars = [
        "sex_simple",
        "race_simple",
        "ethnicity_simple",
        "obese_bmi30",
        "insulin_use_baseline",
        "neuropathy_baseline",
        "nephropathy_baseline",
        "retinopathy_baseline",
        "baseline_back_pain_flag",
    ]

    for arm, g in df.groupby("exposure_group", dropna=False, observed=False):
        arm_name = str(arm)
        rows.append(
            {
                "exposure_group": arm_name,
                "variable": "N",
                "level": "overall",
                "n": int(len(g)),
                "value": float(len(g)),
                "stat": "count",
            }
        )
        for var in numeric_vars:
            if var not in g.columns:
                continue
            non_null = g[var].dropna()
            rows.append(
                {
                    "exposure_group": arm_name,
                    "variable": var,
                    "level": "mean",
                    "n": int(non_null.shape[0]),
                    "value": float(non_null.mean()) if len(non_null) else np.nan,
                    "stat": "mean",
                }
            )
            rows.append(
                {
                    "exposure_group": arm_name,
                    "variable": var,
                    "level": "sd",
                    "n": int(non_null.shape[0]),
                    "value": float(non_null.std(ddof=1)) if len(non_null) > 1 else np.nan,
                    "stat": "sd",
                }
            )
        for var in cat_vars:
            if var not in g.columns:
                continue
            counts = g[var].value_counts(dropna=False)
            for level, cnt in counts.items():
                rows.append(
                    {
                        "exposure_group": arm_name,
                        "variable": var,
                        "level": str(level),
                        "n": int(cnt),
                        "value": float(cnt / len(g)) if len(g) else np.nan,
                        "stat": "proportion",
                    }
                )
    return pd.DataFrame(rows)


def build_severity_baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    if "exposure_group" not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby("exposure_group", dropna=False, observed=False)
    out = grouped.size().rename("n").reset_index()

    metric_map = {
        "hba1c_recent": "hba1c_recent_mean",
        "hba1c_mean_year": "hba1c_mean_year_mean",
        "diabetes_duration_days": "diabetes_duration_days_mean",
        "insulin_use_baseline": "insulin_use_baseline_rate",
        "neuropathy_baseline": "neuropathy_baseline_rate",
        "nephropathy_baseline": "nephropathy_baseline_rate",
        "retinopathy_baseline": "retinopathy_baseline_rate",
        "baseline_condition_count": "baseline_condition_count_mean",
        "baseline_back_pain_flag": "baseline_back_pain_flag_rate",
    }

    for src, dst in metric_map.items():
        if src not in df.columns:
            continue
        stat = grouped[src].mean().rename(dst).reset_index()
        out = out.merge(stat, on="exposure_group", how="left")
    return out


def build_bmi_missingness_table(df: pd.DataFrame, threshold: int, notes: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    overall = pd.DataFrame(
        {
            "stratifier": ["overall"],
            "level": ["overall"],
            "n_total": [int(len(df))],
            "n_bmi_missing": [int(df["bmi_missing"].sum())],
        }
    )
    overall["n_bmi_present"] = overall["n_total"] - overall["n_bmi_missing"]
    overall["missing_rate"] = overall["n_bmi_missing"] / overall["n_total"]
    overall["chi2_p_value"] = np.nan
    rows.append(overall)

    stratifiers = ["exposure_group", "sex_simple", "race_simple", "ethnicity_simple", "age_bin"]
    for strat in stratifiers:
        if strat not in df.columns:
            continue
        tmp = (
            df.groupby(strat, dropna=False, observed=False)
            .agg(n_total=("person_id", "size"), n_bmi_missing=("bmi_missing", "sum"))
            .reset_index()
            .rename(columns={strat: "level"})
        )
        tmp["stratifier"] = strat
        tmp["n_bmi_present"] = tmp["n_total"] - tmp["n_bmi_missing"]
        tmp["missing_rate"] = tmp["n_bmi_missing"] / tmp["n_total"]
        tmp["chi2_p_value"] = np.nan

        if chi2_contingency is not None and len(tmp) >= 2:
            contingency = pd.crosstab(df[strat], df["bmi_missing"], dropna=False)
            if contingency.shape[1] == 2 and int(contingency.values.min()) >= threshold:
                try:
                    _, pval, _, _ = chi2_contingency(contingency)
                    tmp["chi2_p_value"] = pval
                except Exception as exc:  # pragma: no cover
                    notes.append(f"BMI missingness chi-square failed for {strat}: {exc}")
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)


def bmi_missingness_selection_models(
    df: pd.DataFrame,
    config: dict,
    threshold: int,
    notes: list[str],
    model_store: dict[str, object] | None = None,
    diagnostics_store: list[pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_rows: list[pd.DataFrame] = []
    working = df.copy()

    # Crude risk comparison by BMI presence.
    crude = _risk_table_by_group(
        working.loc[working["has_min_followup_2y"] == 1],
        group_col="bmi_present",
        event_col="incident_spine_2y",
    )
    crude["analysis"] = "crude_risk_by_bmi_present"
    out_rows.append(crude)

    # Adjusted logistic selection diagnostic.
    model_df = working.loc[
        (working["has_min_followup_2y"] == 1)
        & working["age_at_index"].notna()
        & working["exposure_group"].notna()
    ].copy()

    base_terms = _base_logit_terms(config)
    formula = "incident_spine_2y ~ bmi_present + " + " + ".join(base_terms)
    adjusted = _fit_logit_if_allowed(
        model_df,
        formula,
        label="bmi_present_adjusted_logit",
        event_col="incident_spine_2y",
        threshold=threshold,
        notes=notes,
        config=config,
        model_store=model_store,
        diagnostics_store=diagnostics_store,
    )
    out_rows.append(adjusted)

    # IPW for BMI-observed process.
    ipw_formula = (
        "bmi_present ~ "
        + " + ".join(base_terms)
        + " + insulin_use_baseline + baseline_condition_count + baseline_outpatient_visits"
    )

    ipw_df = model_df.dropna(
        subset=[
            "bmi_present",
            "age_at_index",
            "insulin_use_baseline",
            "baseline_condition_count",
            "baseline_outpatient_visits",
        ]
    ).copy()

    ipw_result = pd.DataFrame()
    if len(ipw_df) >= threshold:
        try:
            ipw_fit = smf.logit(ipw_formula, data=ipw_df).fit(disp=False)
            if model_store is not None:
                model_store["bmi_observed_ipw_model"] = ipw_fit
            ipw_df["p_obs"] = ipw_fit.predict(ipw_df).clip(1e-4, 1 - 1e-4)
            p_obs_marginal = float(ipw_df["bmi_present"].mean())
            ipw_df["ipw_bmi_observed"] = np.where(
                ipw_df["bmi_present"] == 1,
                p_obs_marginal / ipw_df["p_obs"],
                (1 - p_obs_marginal) / (1 - ipw_df["p_obs"]),
            )
            q1, q99 = ipw_df["ipw_bmi_observed"].quantile([0.01, 0.99]).tolist()
            ipw_df["ipw_bmi_observed_trunc"] = ipw_df["ipw_bmi_observed"].clip(q1, q99)

            ipw_result = _logit_or_table(ipw_fit, "bmi_observed_ipw_model")
            ipw_result["mean_weight"] = ipw_df["ipw_bmi_observed_trunc"].mean()
            ipw_result["max_weight"] = ipw_df["ipw_bmi_observed_trunc"].max()
        except Exception as exc:  # pragma: no cover
            notes.append(f"BMI observed IPW model failed: {exc}")
            ipw_result = pd.DataFrame({"analysis": ["bmi_observed_ipw_model"], "error": [str(exc)]})
    else:
        notes.append("BMI observed IPW model skipped due to small sample.")
        ipw_result = _create_policy_note_df("bmi_observed_ipw_model")

    out_rows.append(ipw_result)

    # Pattern-mixture sensitivity for BMI-missing participants.
    pm_rows: list[pd.DataFrame] = []
    if len(model_df) >= threshold:
        base_complete = model_df.copy()
        group_means = base_complete.groupby("exposure_group", dropna=False, observed=False)["bmi"].transform("mean")
        for delta in (-2.0, 2.0, 5.0):
            tmp = base_complete.copy()
            tmp["bmi_pm"] = tmp["bmi"]
            tmp.loc[tmp["bmi_pm"].isna(), "bmi_pm"] = group_means.loc[tmp["bmi_pm"].isna()] + delta
            tmp["bmi_pm"] = tmp["bmi_pm"].fillna(tmp["bmi"].mean())

            pm_formula = "incident_spine_2y ~ " + " + ".join(base_terms) + " + bmi_pm"
            pm_out = _fit_logit_if_allowed(
                tmp,
                pm_formula,
                label=f"pattern_mixture_delta_{delta:+.1f}",
                event_col="incident_spine_2y",
                threshold=threshold,
                notes=notes,
                config=config,
                model_store=model_store,
                diagnostics_store=diagnostics_store,
            )
            pm_rows.append(pm_out)
    if pm_rows:
        out_rows.extend(pm_rows)

    combined = pd.concat(out_rows, ignore_index=True, sort=False)
    return combined, ipw_df


def run_logistic_models(
    df: pd.DataFrame,
    config: dict,
    threshold: int,
    notes: list[str],
    model_store: dict[str, object] | None = None,
    diagnostics_store: list[pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_2y = bool(config["require_730d_for_binary_models"])
    analysis_df = df.copy()
    if require_2y:
        analysis_df = analysis_df.loc[analysis_df["has_min_followup_2y"] == 1].copy()

    analysis_df = analysis_df.loc[
        analysis_df["age_at_index"].notna()
        & analysis_df["exposure_group"].notna()
        & analysis_df["bmi_c"].notna()
    ].copy()

    exposure_term = _reference_term("exposure_group", config)
    shared_terms = _base_logit_terms(config, include_exposure=False)
    main_formula = "incident_spine_2y ~ " + " + ".join([exposure_term, *shared_terms, "bmi_c"])

    main_out = _fit_logit_if_allowed(
        analysis_df,
        main_formula,
        label="logistic_main_hc3",
        event_col="incident_spine_2y",
        threshold=threshold,
        notes=notes,
        config=config,
        model_store=model_store,
        diagnostics_store=diagnostics_store,
    )

    int_formula = "incident_spine_2y ~ " + " + ".join(
        [f"{exposure_term} * obese_bmi30", *shared_terms, "bmi_c"]
    )

    int_out = _fit_logit_if_allowed(
        analysis_df,
        int_formula,
        label="logistic_interaction_obesity",
        event_col="incident_spine_2y",
        threshold=threshold,
        notes=notes,
        config=config,
        model_store=model_store,
        diagnostics_store=diagnostics_store,
    )

    interaction_rows: list[dict[str, object]] = []
    n_all = int(len(analysis_df))
    events_all = int(analysis_df["incident_spine_2y"].sum()) if n_all else 0
    compliant_all, note_all = model_is_policy_compliant(n_all, events_all, threshold)
    if compliant_all:
        if _should_force_penalized("logistic_interaction_obesity", config):
            interaction_rows.append(
                {
                    "analysis": "interaction_joint_wald",
                    "policy_note": "Joint Wald skipped because logistic_interaction_obesity used penalized estimation.",
                }
            )
        else:
            try:
                fit_int = None
                if model_store is not None:
                    candidate = model_store.get("logistic_interaction_obesity")
                    # Penalized fits do not provide valid Wald covariance for this joint test.
                    if candidate is not None and hasattr(candidate, "wald_test") and hasattr(candidate, "params"):
                        fit_int = candidate
                if fit_int is None:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error", category=ConvergenceWarning)
                        warnings.filterwarnings("error", category=RuntimeWarning)
                        fit_int = smf.logit(formula=int_formula, data=analysis_df).fit(disp=False, cov_type="HC3")
                    if model_store is not None:
                        model_store["logistic_interaction_obesity_joint_wald_fit"] = fit_int
                terms = [
                    f"{exposure_term}[T.glp1_only]:obese_bmi30",
                    f"{exposure_term}[T.combo]:obese_bmi30",
                ]
                present = [t for t in terms if t in fit_int.params.index]
                if len(present) == 2:
                    names = list(fit_int.params.index)
                    r = np.zeros((2, len(names)))
                    r[0, names.index(present[0])] = 1
                    r[1, names.index(present[1])] = 1
                    wald = fit_int.wald_test(r, scalar=True)
                    interaction_rows.append(
                        {
                            "analysis": "interaction_joint_wald",
                            "statistic": float(np.asarray(wald.statistic).reshape(-1)[0]),
                            "p_value": float(wald.pvalue),
                            "df": int(np.asarray(wald.df_denom).reshape(-1)[0]) if hasattr(wald, "df_denom") else np.nan,
                        }
                    )
                else:
                    interaction_rows.append(
                        {
                            "analysis": "interaction_joint_wald",
                            "policy_note": "Interaction terms not fully present in fitted model.",
                        }
                    )
            except Exception as exc:  # pragma: no cover
                notes.append(f"Interaction Wald test failed: {exc}")
                interaction_rows.append({"analysis": "interaction_joint_wald", "error": str(exc)})
    else:
        interaction_rows.append({"analysis": "interaction_joint_wald", "policy_note": note_all})

    # Obesity-stratified models with policy checks.
    for obese_value, obese_label in [(0, "bmi_lt30"), (1, "bmi_ge30")]:
        subset = analysis_df.loc[analysis_df["obese_bmi30"] == obese_value].copy()
        n = int(len(subset))
        events = int(subset["incident_spine_2y"].sum()) if not subset.empty else 0
        compliant, note = model_is_policy_compliant(n, events, threshold)
        if not compliant:
            interaction_rows.append(
                {
                    "analysis": f"stratified_logit_{obese_label}",
                    "policy_note": note,
                }
            )
            continue

        formula = "incident_spine_2y ~ " + " + ".join([exposure_term, *shared_terms, "bmi_c"])
        tab = _fit_logit_if_allowed(
            subset,
            formula,
            label=f"stratified_logit_{obese_label}",
            event_col="incident_spine_2y",
            threshold=threshold,
            notes=notes,
            config=config,
            model_store=model_store,
            diagnostics_store=diagnostics_store,
        )
        interaction_rows.extend(tab.to_dict("records"))

    return main_out, int_out, pd.DataFrame(interaction_rows)


def _prepare_cox_model_df(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    cox_df = df.copy()
    horizon_days = int(config.get("cox_time_horizon_days", config.get("outcome_window_days", 730)))

    if bool(config.get("cox_require_positive_followup_days", True)) and "days_followup" in cox_df.columns:
        cox_df = cox_df.loc[pd.to_numeric(cox_df["days_followup"], errors="coerce") > 0].copy()

    if "person_time_2y_days" in cox_df.columns and "incident_spine_2y" in cox_df.columns:
        cox_df["cox_time_days"] = pd.to_numeric(cox_df["person_time_2y_days"], errors="coerce")
        cox_df["cox_event"] = pd.to_numeric(cox_df["incident_spine_2y"], errors="coerce").fillna(0).astype(int)
    else:
        raw_time = pd.to_numeric(cox_df["time_to_event_or_censor_days"], errors="coerce")
        cox_df["cox_time_days"] = raw_time.clip(upper=horizon_days)
        full_event = pd.to_numeric(cox_df["event_full_followup"], errors="coerce").fillna(0).astype(int)
        cox_df["cox_event"] = np.where((full_event == 1) & (raw_time <= horizon_days), 1, 0)

    cox_df = cox_df.loc[cox_df["cox_time_days"].notna() & (cox_df["cox_time_days"] > 0)].copy()
    return cox_df


def _build_cox_diagnostics(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            [
                {
                    "analysis": label,
                    "scope": "overall",
                    "level": "overall",
                    "n": 0,
                    "events": 0,
                    "nonevents": 0,
                    "event_rate": np.nan,
                    "person_time_days": 0.0,
                }
            ]
        )

    out_rows: list[dict[str, object]] = [
        {
            "analysis": label,
            "scope": "overall",
            "level": "overall",
            "n": int(len(df)),
            "events": int(df["cox_event"].sum()),
            "nonevents": int(len(df) - int(df["cox_event"].sum())),
            "event_rate": float(df["cox_event"].mean()),
            "person_time_days": float(df["cox_time_days"].sum()),
        }
    ]
    if "exposure_group" in df.columns:
        by_group = (
            df.groupby("exposure_group", dropna=False, observed=False)
            .agg(
                n=("cox_event", "size"),
                events=("cox_event", "sum"),
                person_time_days=("cox_time_days", "sum"),
            )
            .reset_index()
        )
        by_group["nonevents"] = by_group["n"] - by_group["events"]
        by_group["event_rate"] = by_group["events"] / by_group["n"]
        for _, row in by_group.iterrows():
            out_rows.append(
                {
                    "analysis": label,
                    "scope": "by_exposure",
                    "level": str(row["exposure_group"]),
                    "n": int(row["n"]),
                    "events": int(row["events"]),
                    "nonevents": int(row["nonevents"]),
                    "event_rate": float(row["event_rate"]) if pd.notna(row["event_rate"]) else np.nan,
                    "person_time_days": float(row["person_time_days"]),
                }
            )
    return pd.DataFrame(out_rows)


def _cox_lifelines(
    df: pd.DataFrame,
    label: str,
    notes: list[str],
    config: dict,
    model_store: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if CoxPHFitter is None:
        return pd.DataFrame(), pd.DataFrame()

    cox_df = df.copy()
    if cox_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    categorical_cols = ["exposure_group", "sex_simple", "race_simple", "ethnicity_simple"]
    cox_df = pd.get_dummies(cox_df, columns=categorical_cols, drop_first=True)

    covars = [
        c
        for c in [
            "age_at_index",
            "bmi",
            "hba1c_recent",
            "diabetes_duration_days",
            "insulin_use_baseline",
            "microvascular_any_baseline",
            "baseline_condition_count",
            "baseline_outpatient_visits",
            "baseline_spine_imaging",
            "baseline_back_pain_flag",
        ]
        if c in cox_df.columns
    ]

    covars.extend([c for c in cox_df.columns if c.startswith("exposure_group_")])
    covars.extend([c for c in cox_df.columns if c.startswith("sex_simple_")])
    covars.extend([c for c in cox_df.columns if c.startswith("race_simple_")])
    covars.extend([c for c in cox_df.columns if c.startswith("ethnicity_simple_")])

    model_df = cox_df[["cox_time_days", "cox_event", *covars]].copy()
    for c in covars:
        if model_df[c].isna().any():
            model_df[c] = model_df[c].fillna(model_df[c].median())

    cph = CoxPHFitter(penalizer=float(config.get("cox_penalizer", 0.0)))
    try:
        cph.fit(
            model_df,
            duration_col="cox_time_days",
            event_col="cox_event",
            robust=True,
        )
        if model_store is not None:
            model_store[label] = cph
        summary = cph.summary.reset_index().rename(columns={"index": "term"})
        out = pd.DataFrame(
            {
                "term": summary["covariate"],
                "coef": summary["coef"],
                "std_error": summary.get("se(coef)", np.nan),
                "hr": summary["exp(coef)"],
                "ci_low": summary["exp(coef) lower 95%"],
                "ci_high": summary["exp(coef) upper 95%"],
                "p_value": summary["p"],
                "model": label,
                "effect_type": "HR",
                "effect_scale": "log",
            }
        )

        ph_df = pd.DataFrame()
        if proportional_hazard_test is not None:
            ph = proportional_hazard_test(cph, model_df, time_transform="rank")
            ph_df = ph.summary.reset_index().rename(columns={"index": "term"})
            ph_df["analysis"] = "ph_assumption_test"
        return out, ph_df
    except Exception as exc:  # pragma: no cover
        notes.append(f"Cox model failed ({label}): {exc}")
        return pd.DataFrame({"analysis": [label], "error": [str(exc)]}), pd.DataFrame()


def _cox_phreg(
    df: pd.DataFrame,
    label: str,
    notes: list[str],
    model_store: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if PHReg is None:
        return pd.DataFrame(), pd.DataFrame()

    cox_df = df.copy()
    if cox_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "person_id" in cox_df.columns:
        groups = pd.to_numeric(cox_df["person_id"], errors="coerce").fillna(-1).to_numpy()
    else:
        groups = np.arange(len(cox_df))

    cox_df = pd.get_dummies(
        cox_df,
        columns=["exposure_group", "sex_simple", "race_simple", "ethnicity_simple"],
        drop_first=True,
    )
    covars = [
        c
        for c in [
            "age_at_index",
            "bmi",
            "hba1c_recent",
            "diabetes_duration_days",
            "insulin_use_baseline",
            "microvascular_any_baseline",
            "baseline_condition_count",
            "baseline_outpatient_visits",
            "baseline_spine_imaging",
            "baseline_back_pain_flag",
        ]
        if c in cox_df.columns
    ]
    covars.extend([c for c in cox_df.columns if c.startswith("exposure_group_")])
    covars.extend([c for c in cox_df.columns if c.startswith("sex_simple_")])
    covars.extend([c for c in cox_df.columns if c.startswith("race_simple_")])
    covars.extend([c for c in cox_df.columns if c.startswith("ethnicity_simple_")])
    covars = list(dict.fromkeys(covars))
    if not covars:
        return pd.DataFrame(), pd.DataFrame()

    design = cox_df[covars].copy()
    for col in covars:
        if design[col].isna().any():
            design[col] = design[col].fillna(design[col].median())
        design[col] = pd.to_numeric(design[col], errors="coerce").fillna(0.0).astype(float)

    variance = design.var(axis=0, ddof=0)
    keep_cols = variance[variance > 0].index.tolist()
    if not keep_cols:
        notes.append(f"PHReg Cox model ({label}) has no non-constant covariates after preprocessing.")
        return pd.DataFrame(), pd.DataFrame()
    if len(keep_cols) < len(covars):
        covars = keep_cols
        design = design[covars]

    endog = pd.to_numeric(cox_df["cox_time_days"], errors="coerce").astype(float).to_numpy()
    status = pd.to_numeric(cox_df["cox_event"], errors="coerce").fillna(0).astype(int).to_numpy()
    exog = design.to_numpy(dtype=float)

    try:
        model = PHReg(endog=endog, exog=exog, status=status, ties="breslow")
        res = model.fit(groups=groups)
        if model_store is not None:
            model_store[label] = res
        conf = res.conf_int()
        out = pd.DataFrame(
            {
                "term": covars,
                "coef": res.params,
                "std_error": getattr(res, "bse", np.repeat(np.nan, len(covars))),
                "hr": np.exp(res.params),
                "ci_low": np.exp(conf[:, 0]),
                "ci_high": np.exp(conf[:, 1]),
                "p_value": res.pvalues,
                "model": label,
                "effect_type": "HR",
                "effect_scale": "log",
            }
        )
        ph_note = pd.DataFrame(
            {
                "analysis": ["ph_assumption_test"],
                "policy_note": ["PH diagnostics limited in PHReg fallback."],
            }
        )
        return out, ph_note
    except Exception as exc:  # pragma: no cover
        notes.append(f"PHReg Cox model failed ({label}): {exc}")
        return pd.DataFrame({"analysis": [label], "error": [str(exc)]}), pd.DataFrame()


def build_km_curve_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    working = _prepare_cox_model_df(df, config)
    if working.empty:
        return pd.DataFrame()

    for arm, g in working.groupby("exposure_group", dropna=False, observed=False):
        if g.empty:
            continue
        times = np.sort(g["cox_time_days"].astype(float).unique())
        n_at_risk = len(g)
        surv = 1.0
        for t in times:
            events = int(((g["cox_time_days"] == t) & (g["cox_event"] == 1)).sum())
            censored = int(((g["cox_time_days"] == t) & (g["cox_event"] == 0)).sum())
            if n_at_risk > 0:
                surv *= (1 - (events / n_at_risk))
            out_rows.append(
                {
                    "exposure_group": str(arm),
                    "time_days": float(t),
                    "n_at_risk": int(n_at_risk),
                    "n_events": int(events),
                    "n_censored": int(censored),
                    "survival_probability": float(surv),
                }
            )
            n_at_risk -= events + censored
            if n_at_risk <= 0:
                break
    return pd.DataFrame(out_rows)


def run_cox_models(
    df: pd.DataFrame,
    config: dict,
    threshold: int,
    notes: list[str],
    model_store: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cox_df = _prepare_cox_model_df(df, config)
    cox_diag = _build_cox_diagnostics(cox_df, "cox_time_to_spine")

    n = int(len(cox_df))
    events = int(cox_df["cox_event"].sum()) if not cox_df.empty else 0
    logging.info("cox_time_to_spine: n=%s events=%s", n, events)
    compliant, note = model_is_policy_compliant(n=n, events=events, threshold=threshold)
    if not compliant:
        notes.append(f"cox_time_to_spine: {note}")
        return _create_policy_note_df("cox_time_to_spine"), pd.DataFrame(), cox_diag, cox_df

    out, ph_df = _cox_lifelines(cox_df, "cox_time_to_spine", notes, config=config, model_store=model_store)
    if out.empty:
        out, ph_df = _cox_phreg(cox_df, "cox_time_to_spine", notes, model_store=model_store)
    if out.empty:
        notes.append(
            "Neither lifelines nor PHReg Cox model succeeded; cox_time_to_spine includes only policy/error notes."
        )
        out = _create_policy_note_df("cox_time_to_spine")
    else:
        out["n_total"] = n
        out["events"] = events
        out["person_time_days"] = float(cox_df["cox_time_days"].sum())

    if not ph_df.empty:
        ph_df = ph_df.rename(columns={"p": "p_value", "test_statistic": "statistic"})
    return out, ph_df, cox_diag, cox_df


def run_additional_interaction_models(
    df: pd.DataFrame,
    config: dict,
    threshold: int,
    notes: list[str],
    model_store: dict[str, object] | None = None,
    diagnostics_store: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    require_2y = bool(config["require_730d_for_binary_models"])
    logistic_df = df.copy()
    if require_2y:
        logistic_df = logistic_df.loc[logistic_df["has_min_followup_2y"] == 1].copy()
    logistic_df = logistic_df.loc[
        logistic_df["bmi_c"].notna()
        & logistic_df["age_at_index"].notna()
        & logistic_df["exposure_group"].notna()
    ].copy()

    exposure_term = _reference_term("exposure_group", config)
    shared_terms = _base_logit_terms(config, include_exposure=False)
    cont_int_formula = "incident_spine_2y ~ " + " + ".join([f"{exposure_term} * bmi_c", *shared_terms])
    rows.append(
        _fit_logit_if_allowed(
            logistic_df,
            cont_int_formula,
            label="logistic_interaction_continuous_bmi",
            event_col="incident_spine_2y",
            threshold=threshold,
            notes=notes,
            config=config,
            model_store=model_store,
            diagnostics_store=diagnostics_store,
        )
    )

    # Cox interactions.
    cox_df = _prepare_cox_model_df(df, config)
    n = int(len(cox_df))
    events = int(cox_df["cox_event"].sum()) if "cox_event" in cox_df.columns else 0
    compliant, note = model_is_policy_compliant(n=n, events=events, threshold=threshold)
    if not compliant:
        rows.append(_create_policy_note_df("cox_interaction_obesity"))
        rows.append(_create_policy_note_df("cox_interaction_continuous_bmi"))
        return pd.concat(rows, ignore_index=True, sort=False)

    if CoxPHFitter is None and PHReg is None:
        notes.append("Neither lifelines nor PHReg is available; Cox interaction models skipped.")
        rows.append(_create_policy_note_df("cox_interaction_obesity"))
        rows.append(_create_policy_note_df("cox_interaction_continuous_bmi"))
        return pd.concat(rows, ignore_index=True, sort=False)

    def _fit_cox_interaction_model(design_df: pd.DataFrame, covars: list[str], model_label: str) -> pd.DataFrame:
        local = design_df[["cox_time_days", "cox_event", *covars]].copy()
        for col in covars:
            local[col] = pd.to_numeric(local[col], errors="coerce").fillna(0.0).astype(float)
        variance = local[covars].var(axis=0, ddof=0)
        covars = variance[variance > 0].index.tolist()
        if not covars:
            return pd.DataFrame({"analysis": [model_label], "error": ["No non-constant covariates"]})
        local = local[["cox_time_days", "cox_event", *covars]]

        if CoxPHFitter is not None:
            try:
                cph = CoxPHFitter(penalizer=float(config.get("cox_penalizer", 0.0)))
                cph.fit(
                    local,
                    duration_col="cox_time_days",
                    event_col="cox_event",
                    robust=True,
                )
                if model_store is not None:
                    model_store[model_label] = cph
                summ = cph.summary.reset_index().rename(columns={"index": "term"})
                return pd.DataFrame(
                    {
                        "term": summ["covariate"],
                        "coef": summ["coef"],
                        "hr": summ["exp(coef)"],
                        "ci_low": summ["exp(coef) lower 95%"],
                        "ci_high": summ["exp(coef) upper 95%"],
                        "p_value": summ["p"],
                        "model": model_label,
                    }
                )
            except Exception as exc:  # pragma: no cover
                notes.append(f"{model_label} lifelines fit failed: {exc}")

        if PHReg is not None:
            try:
                if "person_id" in design_df.columns:
                    groups = pd.to_numeric(design_df["person_id"], errors="coerce").fillna(-1).to_numpy()
                else:
                    groups = np.arange(len(design_df))
                endog = local["cox_time_days"].astype(float).to_numpy()
                status = local["cox_event"].astype(int).to_numpy()
                exog = local[covars].to_numpy(dtype=float)
                ph = PHReg(endog=endog, exog=exog, status=status, ties="breslow")
                res = ph.fit(groups=groups)
                if model_store is not None:
                    model_store[model_label] = res
                conf = res.conf_int()
                return pd.DataFrame(
                    {
                        "term": covars,
                        "coef": res.params,
                        "hr": np.exp(res.params),
                        "ci_low": np.exp(conf[:, 0]),
                        "ci_high": np.exp(conf[:, 1]),
                        "p_value": res.pvalues,
                        "model": model_label,
                    }
                )
            except Exception as exc:  # pragma: no cover
                notes.append(f"{model_label} PHReg fit failed: {exc}")

        return pd.DataFrame({"analysis": [model_label], "error": ["Cox interaction model failed"]})

    if cox_df.empty:
        rows.append(_create_policy_note_df("cox_interaction_obesity"))
        rows.append(_create_policy_note_df("cox_interaction_continuous_bmi"))
        return pd.concat(rows, ignore_index=True, sort=False)

    # Fill missing BMI for Cox interaction models and include missingness indicator.
    cox_df["bmi_missing"] = cox_df["bmi"].isna().astype(int)
    cox_df["bmi_filled"] = cox_df["bmi"].fillna(cox_df["bmi"].median())

    cox_df = pd.get_dummies(
        cox_df,
        columns=["exposure_group", "sex_simple", "race_simple", "ethnicity_simple"],
        drop_first=True,
    )

    exposure_cols = [c for c in cox_df.columns if c.startswith("exposure_group_")]
    base_covars = [
        c
        for c in [
            "age_at_index",
            "bmi_filled",
            "bmi_missing",
            "hba1c_recent",
            "diabetes_duration_days",
            "insulin_use_baseline",
            "microvascular_any_baseline",
            "baseline_condition_count",
            "baseline_outpatient_visits",
            "baseline_spine_imaging",
        ]
        if c in cox_df.columns
    ]
    base_covars.extend([c for c in cox_df.columns if c.startswith("sex_simple_")])
    base_covars.extend([c for c in cox_df.columns if c.startswith("race_simple_")])
    base_covars.extend([c for c in cox_df.columns if c.startswith("ethnicity_simple_")])

    for col in base_covars:
        if cox_df[col].isna().any():
            cox_df[col] = cox_df[col].fillna(cox_df[col].median())

    # Obesity interaction.
    if "obese_bmi30" in cox_df.columns:
        cox_ob = cox_df.copy()
        inter_cols = []
        for col in exposure_cols:
            inter_col = f"{col}:obese_bmi30"
            cox_ob[inter_col] = cox_ob[col] * cox_ob["obese_bmi30"]
            inter_cols.append(inter_col)
        covars_ob = [*base_covars, *exposure_cols, "obese_bmi30", *inter_cols]
        covars_ob = [c for c in covars_ob if c in cox_ob.columns]
        rows.append(_fit_cox_interaction_model(cox_ob, covars_ob, "cox_interaction_obesity"))
    else:
        rows.append(_create_policy_note_df("cox_interaction_obesity"))

    # Continuous BMI interaction.
    cox_cont = cox_df.copy()
    cont_inter_cols = []
    for col in exposure_cols:
        inter_col = f"{col}:bmi_filled"
        cox_cont[inter_col] = cox_cont[col] * cox_cont["bmi_filled"]
        cont_inter_cols.append(inter_col)
    covars_cont = [*base_covars, *exposure_cols, *cont_inter_cols]
    covars_cont = [c for c in covars_cont if c in cox_cont.columns]
    rows.append(_fit_cox_interaction_model(cox_cont, covars_cont, "cox_interaction_continuous_bmi"))

    return pd.concat(rows, ignore_index=True, sort=False)


def _weighted_mean(x: pd.Series, w: pd.Series) -> float:
    return float(np.average(x, weights=w)) if len(x) else np.nan


def _weighted_var(x: pd.Series, w: pd.Series) -> float:
    mu = _weighted_mean(x, w)
    return float(np.average((x - mu) ** 2, weights=w)) if len(x) else np.nan


def _smd_numeric(x_t: pd.Series, x_c: pd.Series, wt_t: pd.Series | None = None, wt_c: pd.Series | None = None) -> float:
    if wt_t is None or wt_c is None:
        m_t, m_c = x_t.mean(), x_c.mean()
        v_t, v_c = x_t.var(ddof=1), x_c.var(ddof=1)
    else:
        m_t, m_c = _weighted_mean(x_t, wt_t), _weighted_mean(x_c, wt_c)
        v_t, v_c = _weighted_var(x_t, wt_t), _weighted_var(x_c, wt_c)
    denom = np.sqrt((v_t + v_c) / 2)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float((m_t - m_c) / denom)


def _ps_numeric_covariates(ps_df: pd.DataFrame) -> list[str]:
    preferred = [
        "age_at_index",
        "age_sq",
        "bmi",
        "bmi_sq",
        "hba1c_recent",
        "hba1c_mean_year",
        "diabetes_duration_days",
        "duration_sq",
        "insulin_use_baseline",
        "microvascular_any_baseline",
        "baseline_condition_count",
        "baseline_outpatient_visits",
        "outpatient_visits_sq",
        "baseline_spine_imaging",
        "baseline_back_pain_flag",
        "duration_x_insulin",
        "age_x_comorbidity",
        "sex_female_x_bmi",
    ]
    return [c for c in preferred if c in ps_df.columns and ps_df[c].notna().any()]


def _ps_categorical_covariates(ps_df: pd.DataFrame) -> list[str]:
    return [c for c in ["sex_simple", "race_simple", "ethnicity_simple"] if c in ps_df.columns]


def _build_ps_formula(ps_df: pd.DataFrame, config: dict) -> tuple[str, list[str]]:
    linear_terms: list[str] = []
    for col in [
        "age_at_index",
        "bmi",
        "hba1c_recent",
        "hba1c_mean_year",
        "diabetes_duration_days",
        "insulin_use_baseline",
        "microvascular_any_baseline",
        "baseline_condition_count",
        "baseline_outpatient_visits",
        "baseline_spine_imaging",
        "baseline_back_pain_flag",
    ]:
        if col in ps_df.columns and ps_df[col].notna().any():
            linear_terms.append(col)
    nonlinear_terms: list[str] = []
    if "age_at_index" in linear_terms:
        nonlinear_terms.append("I(age_at_index ** 2)")
    if "bmi" in linear_terms:
        nonlinear_terms.append("I(bmi ** 2)")
    if "diabetes_duration_days" in linear_terms:
        nonlinear_terms.append("I(diabetes_duration_days ** 2)")
    if "baseline_outpatient_visits" in linear_terms:
        nonlinear_terms.append("I(baseline_outpatient_visits ** 2)")

    interactions: list[str] = []
    if "sex_simple" in ps_df.columns and "bmi" in linear_terms:
        interactions.append(f"{_reference_term('sex_simple', config)}:bmi")
    if "diabetes_duration_days" in linear_terms and "insulin_use_baseline" in linear_terms:
        interactions.append("diabetes_duration_days:insulin_use_baseline")
    if "age_at_index" in linear_terms and "baseline_condition_count" in linear_terms:
        interactions.append("age_at_index:baseline_condition_count")

    cat_terms = [_reference_term(col, config) for col in _ps_categorical_covariates(ps_df)]
    rhs_terms = list(dict.fromkeys([*linear_terms, *nonlinear_terms, *cat_terms, *interactions]))
    formula = "treat_glp1 ~ " + (" + ".join(rhs_terms) if rhs_terms else "1")
    return formula, rhs_terms


def _refresh_imputed_ps_df(ps_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = ps_df.copy()
    if "bmi" in out.columns:
        lo, hi = float(config.get("bmi_min", 10.0)), float(config.get("bmi_max", 80.0))
        out["bmi"] = pd.to_numeric(out["bmi"], errors="coerce").clip(lo, hi)
        bmi_winsor = config.get("bmi_winsor_quantiles", (0.005, 0.995))
        out["bmi"] = _winsorize_series(out["bmi"], float(bmi_winsor[0]), float(bmi_winsor[1]))
        out["obese_bmi30"] = (out["bmi"] >= 30).astype(int)
        out["bmi_c"] = out["bmi"] - float(config.get("bmi_center_value", 30.0))
        out["bmi_sq"] = out["bmi"] ** 2
    for col in ["hba1c_recent", "hba1c_mean_year"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].clip(
                float(config.get("hba1c_plausible_min", 3.0)),
                float(config.get("hba1c_plausible_max", 20.0)),
            )
            hq = config.get("hba1c_winsor_quantiles", (0.005, 0.995))
            out[col] = _winsorize_series(out[col], float(hq[0]), float(hq[1]))
    if "diabetes_duration_days" in out.columns:
        out["diabetes_duration_days"] = pd.to_numeric(out["diabetes_duration_days"], errors="coerce").clip(lower=0)
        out["duration_sq"] = out["diabetes_duration_days"] ** 2
    if "age_at_index" in out.columns:
        out["age_sq"] = pd.to_numeric(out["age_at_index"], errors="coerce") ** 2
    if "baseline_outpatient_visits" in out.columns:
        out["outpatient_visits_sq"] = pd.to_numeric(out["baseline_outpatient_visits"], errors="coerce") ** 2
    if "diabetes_duration_days" in out.columns and "insulin_use_baseline" in out.columns:
        out["duration_x_insulin"] = out["diabetes_duration_days"] * pd.to_numeric(
            out["insulin_use_baseline"], errors="coerce"
        ).fillna(0)
    if "age_at_index" in out.columns and "baseline_condition_count" in out.columns:
        out["age_x_comorbidity"] = out["age_at_index"] * pd.to_numeric(
            out["baseline_condition_count"], errors="coerce"
        ).fillna(0)
    if "sex_simple" in out.columns:
        out["sex_female_flag"] = (out["sex_simple"].astype(str) == "Female").astype(int)
    if "sex_female_flag" in out.columns and "bmi" in out.columns:
        out["sex_female_x_bmi"] = out["sex_female_flag"] * out["bmi"]
    if "microvascular_any_baseline" not in out.columns:
        micro_cols = [c for c in ["neuropathy_baseline", "nephropathy_baseline", "retinopathy_baseline"] if c in out.columns]
        if micro_cols:
            out["microvascular_any_baseline"] = (out[micro_cols].fillna(0).sum(axis=1) > 0).astype(int)
        else:
            out["microvascular_any_baseline"] = 0
    return out


def _build_imputed_ps_datasets(ps_df: pd.DataFrame, config: dict, notes: list[str]) -> list[pd.DataFrame]:
    target_cols = [c for c in config.get("mi_target_columns", ["bmi", "hba1c_recent", "hba1c_mean_year"]) if c in ps_df.columns]
    missing_targets = [c for c in target_cols if ps_df[c].isna().any()]
    if not missing_targets:
        return [_refresh_imputed_ps_df(ps_df, config)]

    m = max(1, int(config.get("mi_num_imputations", 5)))
    max_iter = max(5, int(config.get("mi_max_iter", 20)))
    seed = int(config.get("random_seed", 42))
    base = ps_df.copy()

    predictor_numeric = [
        c
        for c in [
            *target_cols,
            "treat_glp1",
            "incident_spine_2y",
            "event_full_followup",
            "time_to_event_or_censor_days",
            "age_at_index",
            "baseline_condition_count",
            "baseline_outpatient_visits",
            "baseline_spine_imaging",
            "baseline_back_pain_flag",
            "insulin_use_baseline",
            "microvascular_any_baseline",
        ]
        if c in base.columns
    ]
    cat_cols = [c for c in ["exposure_group", "sex_simple", "race_simple", "ethnicity_simple"] if c in base.columns]

    design = pd.DataFrame(index=base.index)
    for col in predictor_numeric:
        design[col] = pd.to_numeric(base[col], errors="coerce")
    if cat_cols:
        dummies = pd.get_dummies(base[cat_cols].astype("category"), prefix=cat_cols, dummy_na=True)
        design = pd.concat([design, dummies], axis=1)
    for col in design.columns:
        if design[col].notna().sum() == 0:
            design[col] = 0.0
        else:
            design[col] = _winsorize_series(design[col], 0.001, 0.999)

    if IterativeImputer is None:
        notes.append("IterativeImputer unavailable; using single stochastic imputation fallback.")
        out = base.copy()
        rng = np.random.default_rng(seed)
        strata_cols = [c for c in ["exposure_group", "sex_simple"] if c in out.columns]
        for col in missing_targets:
            series = pd.to_numeric(out[col], errors="coerce")
            if strata_cols:
                by_mean = out.groupby(strata_cols, observed=False)[col].transform("mean")
                by_std = out.groupby(strata_cols, observed=False)[col].transform("std")
                mu = by_mean.fillna(series.mean())
                sigma = by_std.fillna(series.std(ddof=1))
            else:
                mu = pd.Series(series.mean(), index=out.index)
                sigma = pd.Series(series.std(ddof=1), index=out.index)
            sigma = sigma.fillna(0.0).clip(lower=0.0)
            draws = rng.normal(mu.to_numpy(), sigma.to_numpy())
            miss = series.isna()
            series.loc[miss] = draws[miss]
            out[col] = series.fillna(series.median())
        return [_refresh_imputed_ps_df(out, config)]

    outputs: list[pd.DataFrame] = []
    for i in range(m):
        imputer = IterativeImputer(
            random_state=seed + i,
            sample_posterior=True,
            max_iter=max_iter,
            initial_strategy="median",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            imputed_matrix = imputer.fit_transform(design)
        imputed_df = pd.DataFrame(imputed_matrix, index=design.index, columns=design.columns)
        out = base.copy()
        for col in missing_targets:
            if col in imputed_df.columns:
                out[col] = pd.to_numeric(imputed_df[col], errors="coerce")
        outputs.append(_refresh_imputed_ps_df(out, config))

    notes.append(f"Multiple imputation used {len(outputs)} datasets for: {', '.join(missing_targets)}.")
    return outputs


def _compute_missingness_weights(ps_df: pd.DataFrame, config: dict, notes: list[str]) -> pd.Series:
    weight = pd.Series(1.0, index=ps_df.index, dtype=float, name="missingness_weight")
    if "bmi" not in ps_df.columns:
        return weight

    model_df = ps_df.copy()
    model_df["labs_observed"] = model_df["bmi"].notna().astype(int)
    if "hba1c_recent" in model_df.columns:
        model_df["labs_observed"] = (
            model_df["bmi"].notna() & model_df["hba1c_recent"].notna()
        ).astype(int)

    for col in [
        "age_at_index",
        "baseline_condition_count",
        "baseline_outpatient_visits",
        "baseline_spine_imaging",
        "insulin_use_baseline",
        "microvascular_any_baseline",
    ]:
        if col in model_df.columns:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce").fillna(model_df[col].median())

    rhs_terms = [
        c
        for c in [
            "age_at_index",
            "baseline_condition_count",
            "baseline_outpatient_visits",
            "baseline_spine_imaging",
            "insulin_use_baseline",
            "microvascular_any_baseline",
        ]
        if c in model_df.columns
    ]
    for cat in _ps_categorical_covariates(model_df):
        rhs_terms.append(_reference_term(cat, config))

    formula = "labs_observed ~ " + (" + ".join(rhs_terms) if rhs_terms else "1")
    try:
        fit = smf.glm(formula=formula, data=model_df, family=sm.families.Binomial()).fit()
        p_obs = fit.predict(model_df).clip(1e-4, 1 - 1e-4)
        p_marg = float(model_df["labs_observed"].mean())
        raw = np.where(
            model_df["labs_observed"] == 1,
            p_marg / p_obs,
            (1.0 - p_marg) / (1.0 - p_obs),
        )
        raw_series = pd.Series(raw, index=model_df.index, dtype=float)
        low_q, high_q = config.get("missingness_weight_truncation_quantiles", (0.01, 0.99))
        lo, hi = raw_series.quantile([float(low_q), float(high_q)]).tolist()
        weight = raw_series.clip(lo, hi)
    except Exception as exc:  # pragma: no cover
        notes.append(f"Missingness IPW model failed; using unit weights ({exc}).")

    return weight


def _compute_treatment_weights(
    treat: pd.Series,
    ps: pd.Series,
    *,
    strategy: str,
    stabilized: bool,
) -> pd.Series:
    t = pd.to_numeric(treat, errors="coerce").fillna(0).astype(int)
    e = pd.to_numeric(ps, errors="coerce").clip(1e-4, 1 - 1e-4)
    p_treated = float(t.mean())

    if strategy == "overlap":
        w = np.where(t == 1, 1 - e, e)
    elif strategy == "att":
        w = np.where(t == 1, 1.0, e / (1 - e))
    else:  # stabilized IPTW
        if stabilized:
            w = np.where(t == 1, p_treated / e, (1 - p_treated) / (1 - e))
        else:
            w = np.where(t == 1, 1.0 / e, 1.0 / (1 - e))
    return pd.Series(w, index=treat.index, dtype=float)


def _truncate_weights(weights: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    lo, hi = w.quantile([lower_q, upper_q]).tolist()
    return w.clip(lo, hi)


def _compute_balance_table(
    ps_df: pd.DataFrame,
    *,
    weight_col: str,
    weighting_label: str,
) -> pd.DataFrame:
    treated = ps_df.loc[ps_df["treat_glp1"] == 1].copy()
    control = ps_df.loc[ps_df["treat_glp1"] == 0].copy()
    if treated.empty or control.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    numeric_covars = _ps_numeric_covariates(ps_df)
    for col in numeric_covars:
        t_series = pd.to_numeric(treated[col], errors="coerce")
        c_series = pd.to_numeric(control[col], errors="coerce")
        if t_series.notna().sum() == 0 or c_series.notna().sum() == 0:
            continue
        smd_before = _smd_numeric(t_series.fillna(t_series.median()), c_series.fillna(c_series.median()))
        smd_after = _smd_numeric(
            t_series.fillna(t_series.median()),
            c_series.fillna(c_series.median()),
            wt_t=treated[weight_col],
            wt_c=control[weight_col],
        )
        rows.append(
            {
                "covariate": col,
                "smd_unweighted": smd_before,
                "smd_weighted": smd_after,
                "abs_smd_unweighted": abs(smd_before) if pd.notna(smd_before) else np.nan,
                "abs_smd_weighted": abs(smd_after) if pd.notna(smd_after) else np.nan,
                "n_treated": int(len(treated)),
                "n_control": int(len(control)),
                "weighting_label": weighting_label,
            }
        )

    for cat_col in _ps_categorical_covariates(ps_df):
        dummies = pd.get_dummies(ps_df[cat_col], prefix=cat_col, dummy_na=False)
        for dcol in dummies.columns:
            t_series = dummies.loc[ps_df["treat_glp1"] == 1, dcol]
            c_series = dummies.loc[ps_df["treat_glp1"] == 0, dcol]
            smd_before = _smd_numeric(t_series, c_series)
            smd_after = _smd_numeric(
                t_series,
                c_series,
                wt_t=treated[weight_col],
                wt_c=control[weight_col],
            )
            rows.append(
                {
                    "covariate": dcol,
                    "smd_unweighted": smd_before,
                    "smd_weighted": smd_after,
                    "abs_smd_unweighted": abs(smd_before) if pd.notna(smd_before) else np.nan,
                    "abs_smd_weighted": abs(smd_after) if pd.notna(smd_after) else np.nan,
                    "n_treated": int(len(treated)),
                    "n_control": int(len(control)),
                    "weighting_label": weighting_label,
                }
            )
    return pd.DataFrame(rows)


def _weight_diagnostics_rows(
    weights: pd.Series,
    *,
    label: str,
    n_total: int,
    treat_n: int,
    control_n: int,
) -> pd.DataFrame:
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    rows = [
        {"model": "weight_diagnostics", "term": "mean_weight", "value": float(w.mean())},
        {"model": "weight_diagnostics", "term": "max_weight", "value": float(w.max())},
        {"model": "weight_diagnostics", "term": "p99_weight", "value": float(w.quantile(0.99))},
        {"model": "weight_diagnostics", "term": "p95_weight", "value": float(w.quantile(0.95))},
        {"model": "weight_diagnostics", "term": "min_weight", "value": float(w.min())},
        {"model": "weight_diagnostics", "term": "ess", "value": _effective_sample_size(w)},
    ]
    out = pd.DataFrame(rows)
    out["weighting_label"] = label
    out["n_total"] = int(n_total)
    out["treated_n"] = int(treat_n)
    out["control_n"] = int(control_n)
    return out


def _fit_aipw_binary(
    ps_df: pd.DataFrame,
    *,
    ps_col: str,
    config: dict,
    notes: list[str],
) -> pd.DataFrame:
    rhs = []
    for col in [
        "age_at_index",
        "bmi",
        "hba1c_recent",
        "hba1c_mean_year",
        "diabetes_duration_days",
        "insulin_use_baseline",
        "microvascular_any_baseline",
        "baseline_condition_count",
        "baseline_outpatient_visits",
        "baseline_spine_imaging",
        "baseline_back_pain_flag",
    ]:
        if col in ps_df.columns:
            rhs.append(col)
    for cat in _ps_categorical_covariates(ps_df):
        rhs.append(_reference_term(cat, config))
    formula = "incident_spine_2y ~ treat_glp1" + ((" + " + " + ".join(rhs)) if rhs else "")

    model_df = ps_df.copy()
    for col in _ps_numeric_covariates(model_df):
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce").fillna(model_df[col].median())
    try:
        out_fit = smf.glm(formula=formula, data=model_df, family=sm.families.Binomial()).fit()
        m1_df = model_df.copy()
        m0_df = model_df.copy()
        m1_df["treat_glp1"] = 1
        m0_df["treat_glp1"] = 0
        mu1 = out_fit.predict(m1_df)
        mu0 = out_fit.predict(m0_df)
        y = pd.to_numeric(model_df["incident_spine_2y"], errors="coerce").fillna(0).astype(float)
        t = pd.to_numeric(model_df["treat_glp1"], errors="coerce").fillna(0).astype(int)
        e = pd.to_numeric(model_df[ps_col], errors="coerce").clip(1e-4, 1 - 1e-4)
        influence = mu1 - mu0 + (t * (y - mu1) / e) - ((1 - t) * (y - mu0) / (1 - e))
        psi = float(np.mean(influence))
        se = float(np.std(influence, ddof=1) / np.sqrt(len(influence))) if len(influence) > 1 else np.nan
        z = psi / se if se and np.isfinite(se) and se > 0 else np.nan
        ci_low = psi - 1.96 * se if np.isfinite(se) else np.nan
        ci_high = psi + 1.96 * se if np.isfinite(se) else np.nan
        return pd.DataFrame(
            {
                "term": ["treat_glp1"],
                "coef": [psi],
                "std_error": [se],
                "estimate": [psi],
                "ci_low": [ci_low],
                "ci_high": [ci_high],
                "p_value": [_two_sided_p_from_z(z)],
                "model": ["mi_aipw_ate_risk_difference"],
                "effect_type": ["RD"],
                "effect_scale": ["identity"],
            }
        )
    except Exception as exc:  # pragma: no cover
        notes.append(f"AIPW fit failed: {exc}")
        return pd.DataFrame({"analysis": ["mi_aipw_ate_risk_difference"], "error": [str(exc)]})


def _pool_imputation_effects(effect_rows: pd.DataFrame) -> pd.DataFrame:
    if effect_rows.empty:
        return pd.DataFrame()

    needed = {"model", "term", "coef"}
    if not needed.issubset(effect_rows.columns):
        return effect_rows

    rows: list[dict[str, object]] = []
    group_cols = ["model", "term", "effect_type", "effect_scale"]
    for keys, g in effect_rows.groupby(group_cols, dropna=False):
        model, term, effect_type, effect_scale = keys
        g = g.loc[pd.to_numeric(g["coef"], errors="coerce").notna()].copy()
        if g.empty:
            continue
        m = int(len(g))
        coef = pd.to_numeric(g["coef"], errors="coerce")
        qbar = float(coef.mean())
        se_series = (
            pd.to_numeric(g["std_error"], errors="coerce")
            if "std_error" in g.columns
            else pd.Series(np.nan, index=g.index)
        )
        if se_series.notna().any():
            ubar = float(np.nanmean(np.square(se_series)))
            b = float(np.nanvar(coef, ddof=1)) if m > 1 else 0.0
            tvar = ubar + ((1.0 + (1.0 / m)) * b)
            se_pool = float(np.sqrt(max(tvar, 0.0)))
        else:
            se_pool = np.nan

        if np.isfinite(se_pool) and se_pool > 0:
            z = qbar / se_pool
            ci_low_coef = qbar - 1.96 * se_pool
            ci_high_coef = qbar + 1.96 * se_pool
            p_value = _two_sided_p_from_z(z)
        else:
            ci_low_coef = np.nan
            ci_high_coef = np.nan
            p_value = np.nan

        row: dict[str, object] = {
            "model": model,
            "term": term,
            "coef": qbar,
            "std_error": se_pool,
            "p_value": p_value,
            "effect_type": effect_type,
            "effect_scale": effect_scale,
            "imputations": m,
            "n_total": float(pd.to_numeric(g.get("n_total", np.nan), errors="coerce").mean()),
            "events": float(pd.to_numeric(g.get("events", np.nan), errors="coerce").mean()),
        }
        if effect_scale == "log":
            row["ci_low"] = float(np.exp(ci_low_coef)) if np.isfinite(ci_low_coef) else np.nan
            row["ci_high"] = float(np.exp(ci_high_coef)) if np.isfinite(ci_high_coef) else np.nan
            if effect_type == "OR":
                row["or"] = float(np.exp(qbar))
            elif effect_type == "HR":
                row["hr"] = float(np.exp(qbar))
            row["estimate"] = float(np.exp(qbar))
        else:
            row["ci_low"] = ci_low_coef
            row["ci_high"] = ci_high_coef
            row["estimate"] = qbar
        rows.append(row)
    return pd.DataFrame(rows)


def _pool_balance_tables(balance_frames: list[pd.DataFrame], target_abs_smd: float) -> pd.DataFrame:
    if not balance_frames:
        return pd.DataFrame()
    bal = pd.concat(balance_frames, ignore_index=True, sort=False)
    if bal.empty:
        return bal
    pooled = (
        bal.groupby(["covariate", "weighting_label"], dropna=False)
        .agg(
            smd_unweighted=("smd_unweighted", "mean"),
            smd_weighted=("smd_weighted", "mean"),
            abs_smd_unweighted=("abs_smd_unweighted", "mean"),
            abs_smd_weighted=("abs_smd_weighted", "mean"),
            n_treated=("n_treated", "mean"),
            n_control=("n_control", "mean"),
        )
        .reset_index()
    )
    summary_rows: list[dict[str, object]] = []
    for label, g in pooled.groupby("weighting_label", dropna=False):
        max_abs = float(pd.to_numeric(g["abs_smd_weighted"], errors="coerce").max())
        summary_rows.append(
            {
                "covariate": "__summary_max_abs_smd__",
                "weighting_label": label,
                "smd_unweighted": np.nan,
                "smd_weighted": np.nan,
                "abs_smd_unweighted": np.nan,
                "abs_smd_weighted": max_abs,
                "n_treated": float(pd.to_numeric(g["n_treated"], errors="coerce").mean()),
                "n_control": float(pd.to_numeric(g["n_control"], errors="coerce").mean()),
                "balance_target_abs_smd": target_abs_smd,
                "balance_pass": bool(max_abs < target_abs_smd) if np.isfinite(max_abs) else False,
            }
        )
    if summary_rows:
        pooled = pd.concat([pooled, pd.DataFrame(summary_rows)], ignore_index=True, sort=False)
    return pooled


def build_ps_and_outcomes(
    df: pd.DataFrame,
    config: dict,
    threshold: int,
    notes: list[str],
    model_store: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    control_label = str(config.get("reference_levels", {}).get("exposure_group", "metformin_only"))
    treated_label = "glp1_only"
    contrast_label = f"{treated_label}_vs_{control_label}"
    target_abs_smd = float(config.get("ps_balance_target_abs_smd", 0.10))

    ps_df = df.loc[df["exposure_group"].isin([control_label, treated_label])].copy()
    if bool(config["require_730d_for_binary_models"]):
        ps_df = ps_df.loc[ps_df["has_min_followup_2y"] == 1].copy()
    if ps_df.empty:
        return _create_policy_note_df("propensity_score_results"), _create_policy_note_df("balance_diagnostics")

    ps_df["treat_glp1"] = (ps_df["exposure_group"] == treated_label).astype(int)
    needed = [
        "incident_spine_2y",
        "time_to_event_or_censor_days",
        "event_full_followup",
        "treat_glp1",
        "exposure_group",
        *set(_ps_numeric_covariates(ps_df)),
        *set(_ps_categorical_covariates(ps_df)),
    ]
    needed = [c for c in dict.fromkeys(needed) if c in ps_df.columns]
    ps_df = ps_df[needed].copy()

    for cat in _ps_categorical_covariates(ps_df) + (["exposure_group"] if "exposure_group" in ps_df.columns else []):
        ps_df[cat] = ps_df[cat].astype("category")

    n = int(len(ps_df))
    events = int(pd.to_numeric(ps_df.get("incident_spine_2y", 0), errors="coerce").fillna(0).sum())
    compliant, note = model_is_policy_compliant(n, events, threshold)
    if not compliant:
        notes.append(f"propensity_score_results: {note}")
        return _create_policy_note_df("propensity_score_results"), _create_policy_note_df("balance_diagnostics")

    missingness_weight = _compute_missingness_weights(ps_df, config, notes=notes)
    imputed_sets = _build_imputed_ps_datasets(ps_df, config, notes=notes)

    effect_rows: list[pd.DataFrame] = []
    diagnostics_rows: list[pd.DataFrame] = []
    balance_frames: list[pd.DataFrame] = []

    strategies = [str(x) for x in config.get("ps_weighting_strategies", ["overlap", "iptw", "att"])]
    truncation_options = config.get("ps_weight_truncation_options")
    if not truncation_options:
        truncation_options = [config.get("ps_weight_truncation_quantiles", (0.01, 0.99))]
    truncation_options = [(float(x[0]), float(x[1])) for x in truncation_options]

    for imp_idx, imp_df in enumerate(imputed_sets, start=1):
        work = imp_df.copy()
        for col in _ps_numeric_covariates(work) + [
            "incident_spine_2y",
            "event_full_followup",
            "time_to_event_or_censor_days",
            "treat_glp1",
        ]:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
                if work[col].isna().any():
                    work[col] = work[col].fillna(work[col].median())

        ps_candidates: dict[str, pd.Series] = {}
        formula, _ = _build_ps_formula(work, config)
        try:
            ps_fit = smf.glm(formula=formula, data=work, family=sm.families.Binomial()).fit()
            if model_store is not None:
                model_store[f"propensity_model_{contrast_label}_imp{imp_idx}_glm"] = ps_fit
            ps_tab = _glm_or_table(ps_fit, f"propensity_model_{contrast_label}_glm")
            ps_tab["imputation_id"] = imp_idx
            ps_tab["n_total"] = len(work)
            ps_tab["treated_n"] = int(work["treat_glp1"].sum())
            ps_tab["control_n"] = int((1 - work["treat_glp1"]).sum())
            effect_rows.append(ps_tab)
            ps_candidates["formula_glm"] = pd.Series(ps_fit.predict(work), index=work.index)
        except Exception as exc_glm:  # pragma: no cover
            notes.append(f"Imputation {imp_idx}: PS GLM failed ({exc_glm}); trying logit.")
            try:
                ps_fit = smf.logit(formula=formula, data=work).fit(disp=False)
                if model_store is not None:
                    model_store[f"propensity_model_{contrast_label}_imp{imp_idx}_logit"] = ps_fit
                ps_tab = _logit_or_table(ps_fit, f"propensity_model_{contrast_label}_logit")
                ps_tab["imputation_id"] = imp_idx
                ps_tab["n_total"] = len(work)
                ps_tab["treated_n"] = int(work["treat_glp1"].sum())
                ps_tab["control_n"] = int((1 - work["treat_glp1"]).sum())
                effect_rows.append(ps_tab)
                ps_candidates["formula_logit"] = pd.Series(ps_fit.predict(work), index=work.index)
            except Exception as exc_logit:  # pragma: no cover
                notes.append(f"Imputation {imp_idx}: PS logit failed ({exc_logit}).")

        if bool(config.get("ps_use_ml_if_available", False)) and GradientBoostingClassifier is not None:
            try:
                ml_covars = [c for c in _ps_numeric_covariates(work) if c in work.columns]
                x = pd.get_dummies(work[[*ml_covars, *_ps_categorical_covariates(work)]], dummy_na=True)
                y = work["treat_glp1"].astype(int)
                gbt = GradientBoostingClassifier(
                    n_estimators=int(config.get("ps_ml_n_estimators", 300)),
                    learning_rate=float(config.get("ps_ml_learning_rate", 0.05)),
                    subsample=float(config.get("ps_ml_subsample", 0.8)),
                    random_state=int(config.get("random_seed", 42)) + imp_idx,
                )
                gbt.fit(x, y)
                ps_ml = pd.Series(gbt.predict_proba(x)[:, 1], index=work.index)
                ps_candidates["gradient_boosting"] = ps_ml
            except Exception as exc_ml:  # pragma: no cover
                notes.append(f"Imputation {imp_idx}: ML PS candidate skipped ({exc_ml}).")

        if not ps_candidates:
            continue

        best: dict[str, object] | None = None
        min_ess_ratio = float(config.get("ps_min_ess_ratio", 0.30))
        min_ess = float(len(work)) * min_ess_ratio
        for ps_name, ps_pred in ps_candidates.items():
            ps_clean = pd.to_numeric(ps_pred, errors="coerce").clip(1e-4, 1 - 1e-4)
            for strategy in strategies:
                raw_w = _compute_treatment_weights(
                    work["treat_glp1"],
                    ps_clean,
                    strategy=strategy,
                    stabilized=bool(config.get("ps_stabilized_weights", True)),
                )
                for q_lo, q_hi in truncation_options:
                    w_treat = _truncate_weights(raw_w, q_lo, q_hi)
                    label = f"{ps_name}|{strategy}|q{q_lo:.3f}-{q_hi:.3f}"
                    trial = work.copy()
                    trial["w_treat"] = w_treat
                    bal = _compute_balance_table(trial, weight_col="w_treat", weighting_label=label)
                    if bal.empty:
                        continue
                    max_abs = float(pd.to_numeric(bal["abs_smd_weighted"], errors="coerce").max())
                    ess = _effective_sample_size(w_treat)
                    if not np.isfinite(ess) or ess < min_ess:
                        continue
                    if best is None or (
                        max_abs < float(best["max_abs_smd"])
                        or (
                            np.isclose(max_abs, float(best["max_abs_smd"]))
                            and ess > float(best["ess"])
                        )
                    ):
                        best = {
                            "ps_name": ps_name,
                            "strategy": strategy,
                            "label": label,
                            "ps": ps_clean,
                            "w_treat": w_treat,
                            "max_abs_smd": max_abs,
                            "ess": ess,
                            "balance": bal,
                        }

        if best is None:
            notes.append(
                f"Imputation {imp_idx}: no PS candidate met ESS floor ({min_ess_ratio:.2f}); relaxing to best SMD."
            )
            for ps_name, ps_pred in ps_candidates.items():
                ps_clean = pd.to_numeric(ps_pred, errors="coerce").clip(1e-4, 1 - 1e-4)
                for strategy in strategies:
                    raw_w = _compute_treatment_weights(
                        work["treat_glp1"],
                        ps_clean,
                        strategy=strategy,
                        stabilized=bool(config.get("ps_stabilized_weights", True)),
                    )
                    for q_lo, q_hi in truncation_options:
                        w_treat = _truncate_weights(raw_w, q_lo, q_hi)
                        label = f"{ps_name}|{strategy}|q{q_lo:.3f}-{q_hi:.3f}"
                        trial = work.copy()
                        trial["w_treat"] = w_treat
                        bal = _compute_balance_table(trial, weight_col="w_treat", weighting_label=label)
                        if bal.empty:
                            continue
                        max_abs = float(pd.to_numeric(bal["abs_smd_weighted"], errors="coerce").max())
                        ess = _effective_sample_size(w_treat)
                        if best is None or max_abs < float(best["max_abs_smd"]):
                            best = {
                                "ps_name": ps_name,
                                "strategy": strategy,
                                "label": label,
                                "ps": ps_clean,
                                "w_treat": w_treat,
                                "max_abs_smd": max_abs,
                                "ess": ess,
                                "balance": bal,
                            }
        if best is None:
            continue

        work["ps_selected"] = pd.to_numeric(best["ps"], errors="coerce").clip(1e-4, 1 - 1e-4)
        work["w_treat"] = pd.to_numeric(best["w_treat"], errors="coerce").fillna(0.0)
        work["w_missing"] = pd.to_numeric(missingness_weight.reindex(work.index), errors="coerce").fillna(1.0)
        work["w_double"] = work["w_treat"] * work["w_missing"]

        if model_store is not None:
            model_store[f"ps_selected_weights_imp{imp_idx}"] = {
                "ps_source": best["ps_name"],
                "strategy": best["strategy"],
                "label": best["label"],
                "max_abs_smd": float(best["max_abs_smd"]),
                "ess": float(best["ess"]),
            }

        bal_primary = _compute_balance_table(work, weight_col="w_treat", weighting_label=f"{best['label']}|primary")
        bal_primary["imputation_id"] = imp_idx
        balance_frames.append(bal_primary)
        bal_double = _compute_balance_table(work, weight_col="w_double", weighting_label=f"{best['label']}|double")
        bal_double["imputation_id"] = imp_idx
        balance_frames.append(bal_double)

        diagnostics_rows.append(
            _weight_diagnostics_rows(
                work["w_treat"],
                label=f"{best['label']}|primary",
                n_total=len(work),
                treat_n=int(work["treat_glp1"].sum()),
                control_n=int((1 - work["treat_glp1"]).sum()),
            ).assign(imputation_id=imp_idx)
        )
        diagnostics_rows.append(
            _weight_diagnostics_rows(
                work["w_double"],
                label=f"{best['label']}|double",
                n_total=len(work),
                treat_n=int(work["treat_glp1"].sum()),
                control_n=int((1 - work["treat_glp1"]).sum()),
            ).assign(imputation_id=imp_idx)
        )

        # Primary weighted logistic model.
        try:
            out_fit = smf.glm(
                "incident_spine_2y ~ treat_glp1",
                data=work,
                family=sm.families.Binomial(),
                freq_weights=work["w_treat"],
            ).fit(cov_type="HC3")
            if model_store is not None:
                model_store[f"mi_weighted_logistic_primary_imp{imp_idx}"] = out_fit
            out_tab = _glm_or_table(out_fit, "mi_weighted_logistic_primary")
            out_tab["imputation_id"] = imp_idx
            out_tab["n_total"] = len(work)
            out_tab["events"] = int(work["incident_spine_2y"].sum())
            out_tab["weighting_label"] = f"{best['label']}|primary"
            effect_rows.append(out_tab)
        except Exception as exc:  # pragma: no cover
            notes.append(f"Imputation {imp_idx}: primary weighted logistic failed ({exc}).")

        # Doubly robust weighted logistic.
        dr_terms = [
            c
            for c in [
                "age_at_index",
                "bmi",
                "hba1c_recent",
                "hba1c_mean_year",
                "diabetes_duration_days",
                "insulin_use_baseline",
                "microvascular_any_baseline",
                "baseline_condition_count",
                "baseline_outpatient_visits",
                "baseline_spine_imaging",
                "baseline_back_pain_flag",
            ]
            if c in work.columns
        ]
        dr_terms.extend(_reference_term(c, config) for c in _ps_categorical_covariates(work))
        dr_rhs = " + ".join(dr_terms)
        dr_formula = "incident_spine_2y ~ treat_glp1" + (f" + {dr_rhs}" if dr_rhs else "")
        try:
            dr_fit = smf.glm(
                dr_formula,
                data=work,
                family=sm.families.Binomial(),
                freq_weights=work["w_treat"],
            ).fit(cov_type="HC3")
            if model_store is not None:
                model_store[f"mi_doubly_robust_weighted_logit_imp{imp_idx}"] = dr_fit
            dr_tab = _glm_or_table(dr_fit, "mi_doubly_robust_weighted_logit")
            dr_tab["imputation_id"] = imp_idx
            dr_tab["n_total"] = len(work)
            dr_tab["events"] = int(work["incident_spine_2y"].sum())
            dr_tab["weighting_label"] = f"{best['label']}|primary"
            effect_rows.append(dr_tab)
        except Exception as exc:  # pragma: no cover
            notes.append(f"Imputation {imp_idx}: doubly robust weighted logit failed ({exc}).")

        # Double-weighting sensitivity.
        try:
            dbl_fit = smf.glm(
                "incident_spine_2y ~ treat_glp1",
                data=work,
                family=sm.families.Binomial(),
                freq_weights=work["w_double"],
            ).fit(cov_type="HC3")
            if model_store is not None:
                model_store[f"mi_weighted_logistic_double_imp{imp_idx}"] = dbl_fit
            dbl_tab = _glm_or_table(dbl_fit, "mi_weighted_logistic_double_weighted")
            dbl_tab["imputation_id"] = imp_idx
            dbl_tab["n_total"] = len(work)
            dbl_tab["events"] = int(work["incident_spine_2y"].sum())
            dbl_tab["weighting_label"] = f"{best['label']}|double"
            effect_rows.append(dbl_tab)
        except Exception as exc:  # pragma: no cover
            notes.append(f"Imputation {imp_idx}: double-weighted logistic failed ({exc}).")

        aipw_tab = _fit_aipw_binary(work, ps_col="ps_selected", config=config, notes=notes)
        if not aipw_tab.empty and "coef" in aipw_tab.columns:
            aipw_tab["imputation_id"] = imp_idx
            aipw_tab["n_total"] = len(work)
            aipw_tab["events"] = int(work["incident_spine_2y"].sum())
            aipw_tab["weighting_label"] = f"{best['label']}|primary"
            effect_rows.append(aipw_tab)

        if CoxPHFitter is not None:
            cox_df = work[
                ["time_to_event_or_censor_days", "event_full_followup", "treat_glp1", "w_treat"]
            ].copy()
            cox_df = cox_df.loc[
                cox_df["time_to_event_or_censor_days"].notna() & (cox_df["time_to_event_or_censor_days"] > 0)
            ]
            cox_events = int(cox_df["event_full_followup"].sum()) if not cox_df.empty else 0
            if len(cox_df) >= threshold and cox_events >= threshold:
                try:
                    cph = CoxPHFitter(penalizer=float(config.get("cox_penalizer", 0.0)))
                    cph.fit(
                        cox_df,
                        duration_col="time_to_event_or_censor_days",
                        event_col="event_full_followup",
                        weights_col="w_treat",
                        robust=True,
                    )
                    if model_store is not None:
                        model_store[f"mi_weighted_cox_imp{imp_idx}"] = cph
                    csum = cph.summary.reset_index()
                    ctab = pd.DataFrame(
                        {
                            "term": csum["covariate"],
                            "coef": csum["coef"],
                            "std_error": csum.get("se(coef)", np.nan),
                            "hr": csum["exp(coef)"],
                            "ci_low": csum["exp(coef) lower 95%"],
                            "ci_high": csum["exp(coef) upper 95%"],
                            "p_value": csum["p"],
                            "model": "mi_weighted_cox_primary",
                            "effect_type": "HR",
                            "effect_scale": "log",
                            "imputation_id": imp_idx,
                            "n_total": len(cox_df),
                            "events": cox_events,
                            "weighting_label": f"{best['label']}|primary",
                        }
                    )
                    effect_rows.append(ctab)
                except Exception as exc:  # pragma: no cover
                    notes.append(f"Imputation {imp_idx}: weighted Cox failed ({exc}).")
        else:
            notes.append("lifelines not available; MI weighted Cox skipped.")

    if effect_rows:
        effect_df = pd.concat(effect_rows, ignore_index=True, sort=False)
        pooled_effects = _pool_imputation_effects(effect_df)
    else:
        pooled_effects = pd.DataFrame()

    if diagnostics_rows:
        diag_df = pd.concat(diagnostics_rows, ignore_index=True, sort=False)
        diag_summary = (
            diag_df.groupby(["model", "term", "weighting_label"], dropna=False)
            .agg(
                value=("value", "mean"),
                value_max=("value", "max"),
                value_min=("value", "min"),
                imputations=("imputation_id", "nunique"),
                n_total=("n_total", "mean"),
                treated_n=("treated_n", "mean"),
                control_n=("control_n", "mean"),
            )
            .reset_index()
        )
        diag_summary["analysis"] = "weight_diagnostics"
    else:
        diag_summary = pd.DataFrame()

    if not pooled_effects.empty and not diag_summary.empty:
        ps_results = pd.concat([pooled_effects, diag_summary], ignore_index=True, sort=False)
    elif not pooled_effects.empty:
        ps_results = pooled_effects
    elif not diag_summary.empty:
        ps_results = diag_summary
    else:
        ps_results = _create_policy_note_df("propensity_score_results")

    balance = _pool_balance_tables(balance_frames, target_abs_smd=target_abs_smd)
    if balance.empty:
        balance = _create_policy_note_df("balance_diagnostics")

    summary_max = pd.to_numeric(
        balance.loc[balance["covariate"] == "__summary_max_abs_smd__", "abs_smd_weighted"], errors="coerce"
    )
    if summary_max.notna().any():
        best_smd = float(summary_max.min())
        notes.append(
            f"Best post-weight max |SMD| across MI sets: {best_smd:.3f} (target < {target_abs_smd:.2f})."
        )
    return ps_results, balance


def utilization_bias_analysis(
    df: pd.DataFrame,
    config: dict,
    threshold: int,
    notes: list[str],
    model_store: dict[str, object] | None = None,
    diagnostics_store: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    util_summary = (
        df.groupby("exposure_group", dropna=False, observed=False)
        .agg(
            n=("person_id", "size"),
            baseline_outpatient_visits_mean=("baseline_outpatient_visits", "mean"),
            baseline_spine_imaging_mean=("baseline_spine_imaging", "mean"),
            baseline_endocrinology_visits_mean=("baseline_endocrinology_visits", "mean"),
            baseline_orthopedics_visits_mean=("baseline_orthopedics_visits", "mean"),
        )
        .reset_index()
    )
    util_summary["analysis"] = "utilization_descriptives"
    rows.append(util_summary)

    model_df = df.loc[
        (df["has_min_followup_2y"] == 1)
        & df["age_at_index"].notna()
        & df["bmi_c"].notna()
        & df["exposure_group"].notna()
    ].copy()
    exposure_term = _reference_term("exposure_group", config)
    shared_terms = _base_logit_terms(config, include_exposure=False)
    formula = "incident_spine_2y ~ " + " + ".join(
        [
            exposure_term,
            *shared_terms,
            "bmi_c",
            "baseline_outpatient_visits",
            "baseline_spine_imaging",
            "baseline_endocrinology_visits",
            "baseline_orthopedics_visits",
        ]
    )

    util_model = _fit_logit_if_allowed(
        model_df,
        formula,
        label="utilization_adjusted_logit",
        event_col="incident_spine_2y",
        threshold=threshold,
        notes=notes,
        config=config,
        model_store=model_store,
        diagnostics_store=diagnostics_store,
    )
    rows.append(util_model)

    # Sensitivity: restrict within utilization tertiles.
    tertile_df = model_df.copy()
    try:
        tertile_df["utilization_tertile"] = pd.qcut(
            tertile_df["baseline_outpatient_visits"].rank(method="first"),
            q=3,
            labels=["low", "mid", "high"],
        )
        for tertile, sub in tertile_df.groupby("utilization_tertile", dropna=False, observed=False):
            n = int(len(sub))
            events = int(sub["incident_spine_2y"].sum())
            compliant, note = model_is_policy_compliant(n, events, threshold)
            if not compliant:
                rows.append(pd.DataFrame({"analysis": [f"utilization_tertile_{tertile}"], "policy_note": [note]}))
                continue
            tertile_formula = "incident_spine_2y ~ " + " + ".join([exposure_term, *shared_terms, "bmi_c"])
            tab = _fit_logit_if_allowed(
                sub,
                tertile_formula,
                label=f"utilization_tertile_{tertile}",
                event_col="incident_spine_2y",
                threshold=threshold,
                notes=notes,
                config=config,
                model_store=model_store,
                diagnostics_store=diagnostics_store,
            )
            rows.append(tab)
    except Exception as exc:  # pragma: no cover
        notes.append(f"Utilization tertile sensitivity failed: {exc}")
        rows.append(pd.DataFrame({"analysis": ["utilization_tertile_sensitivity"], "error": [str(exc)]}))

    return pd.concat(rows, ignore_index=True, sort=False)


def _closest_measurement_per_window(
    person_df: pd.DataFrame,
    target_days: list[int],
    tolerance_days: int,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    if person_df.empty:
        for day in target_days:
            out[f"bmi_{day}d"] = np.nan
        return out

    value_col = "bmi"
    if value_col not in person_df.columns:
        if "bmi_post" in person_df.columns:
            value_col = "bmi_post"
        else:
            for day in target_days:
                out[f"bmi_{day}d"] = np.nan
            return out

    for day in target_days:
        target_date = person_df["index_date"].iloc[0] + pd.Timedelta(days=day)
        tmp = person_df.copy()
        tmp["abs_diff"] = (tmp["measurement_date"] - target_date).abs().dt.days
        tmp = tmp.loc[tmp["abs_diff"] <= tolerance_days]
        if tmp.empty:
            out[f"bmi_{day}d"] = np.nan
        else:
            out[f"bmi_{day}d"] = float(tmp.sort_values("abs_diff").iloc[0][value_col])
    return out


def weight_change_analysis(
    baseline_df: pd.DataFrame,
    post_index_bmi_df: pd.DataFrame,
    config: dict,
    threshold: int,
    notes: list[str],
    model_store: dict[str, object] | None = None,
) -> pd.DataFrame:
    if post_index_bmi_df.empty:
        notes.append("Weight-change analysis skipped: no post-index BMI rows returned.")
        return _create_policy_note_df("weight_change_analysis")

    target_days = list(config["weight_change_target_days"])
    tolerance = int(config["weight_change_window_tolerance_days"])

    merged = post_index_bmi_df.merge(
        baseline_df[["person_id", "index_date", "exposure_group", "bmi", "incident_spine_2y"]],
        on=["person_id", "index_date", "exposure_group"],
        how="inner",
        suffixes=("_post", "_baseline"),
    )
    if merged.empty:
        notes.append("Weight-change analysis skipped: no overlap between baseline cohort and post-index BMI measurements.")
        return _create_policy_note_df("weight_change_analysis")

    rows: list[dict[str, object]] = []
    for person_id, g in merged.groupby("person_id", dropna=False):
        window_vals = _closest_measurement_per_window(g, target_days, tolerance)
        record: dict[str, object] = {
            "person_id": person_id,
            "exposure_group": g["exposure_group"].iloc[0],
            "index_date": g["index_date"].iloc[0],
            "bmi_baseline": g["bmi_baseline"].iloc[0],
            "incident_spine_2y": g["incident_spine_2y"].iloc[0],
        }
        record.update(window_vals)

        # Slope if at least 2 points.
        points = []
        for day in target_days:
            val = record.get(f"bmi_{day}d")
            if pd.notna(val):
                points.append((day, float(val)))
        if len(points) >= 2:
            xs = np.array([p[0] for p in points], dtype=float)
            ys = np.array([p[1] for p in points], dtype=float)
            slope = np.polyfit(xs, ys, deg=1)[0]
            record["bmi_slope_per_day"] = float(slope)
        else:
            record["bmi_slope_per_day"] = np.nan

        rows.append(record)

    traj = pd.DataFrame(rows)
    for day in target_days:
        col = f"bmi_{day}d"
        traj[f"delta_{day}d"] = traj[col] - traj["bmi_baseline"]

    summary_rows: list[pd.DataFrame] = []
    agg_spec: dict[str, tuple[str, str]] = {
        "n": ("person_id", "size"),
        "bmi_slope_per_day_mean": ("bmi_slope_per_day", "mean"),
    }
    for day in target_days:
        dcol = f"delta_{day}d"
        if dcol in traj.columns:
            agg_spec[f"{dcol}_mean"] = (dcol, "mean")

    summary = traj.groupby("exposure_group", dropna=False).agg(**agg_spec).reset_index()
    summary["analysis"] = "weight_change_descriptives"
    summary_rows.append(summary)

    # Exploratory mediation-ish model: outcome on treatment + delta BMI at 12 months.
    preferred_delta = "delta_365d" if "delta_365d" in traj.columns else f"delta_{target_days[min(1, len(target_days)-1)]}d"
    model_df = traj.loc[
        traj[preferred_delta].notna() & traj["exposure_group"].notna() & traj["incident_spine_2y"].notna()
    ].copy()
    n = int(len(model_df))
    events = int(model_df["incident_spine_2y"].sum()) if not model_df.empty else 0
    compliant, note = model_is_policy_compliant(n, events, threshold)
    if compliant:
        try:
            exposure_term = _reference_term("exposure_group", config)
            fit = smf.logit(
                f"incident_spine_2y ~ {exposure_term} + {preferred_delta}",
                data=model_df,
            ).fit(disp=False, cov_type="HC3")
            if model_store is not None:
                model_store[f"weight_change_{preferred_delta}_logit"] = fit
            summary_rows.append(_logit_or_table(fit, f"weight_change_{preferred_delta}_logit"))
        except Exception as exc:  # pragma: no cover
            notes.append(f"Weight-change exploratory model failed: {exc}")
            summary_rows.append(pd.DataFrame({"analysis": [f"weight_change_{preferred_delta}_logit"], "error": [str(exc)]}))
    else:
        summary_rows.append(_create_policy_note_df(f"weight_change_{preferred_delta}_logit"))

    return pd.concat(summary_rows, ignore_index=True, sort=False)


def build_forest_ready(
    logistic_main: pd.DataFrame,
    cox_results: pd.DataFrame,
    ps_results: pd.DataFrame,
    utilization_results: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if not logistic_main.empty and {"term", "or", "ci_low", "ci_high", "model"}.issubset(logistic_main.columns):
        tmp = logistic_main[["term", "or", "ci_low", "ci_high", "model"]].copy()
        tmp = tmp.rename(columns={"or": "estimate"})
        tmp["effect_type"] = "OR"
        frames.append(tmp)

    if not cox_results.empty and {"term", "hr", "ci_low", "ci_high", "model"}.issubset(cox_results.columns):
        tmp = cox_results[["term", "hr", "ci_low", "ci_high", "model"]].copy()
        tmp = tmp.rename(columns={"hr": "estimate"})
        tmp["effect_type"] = "HR"
        frames.append(tmp)

    if not ps_results.empty and "model" in ps_results.columns:
        if {"term", "or", "ci_low", "ci_high"}.issubset(ps_results.columns):
            tmp = ps_results[["term", "or", "ci_low", "ci_high", "model"]].copy()
            tmp = tmp.rename(columns={"or": "estimate"})
            tmp["effect_type"] = "OR"
            frames.append(tmp)
        if {"term", "hr", "ci_low", "ci_high"}.issubset(ps_results.columns):
            tmp = ps_results[["term", "hr", "ci_low", "ci_high", "model"]].copy()
            tmp = tmp.rename(columns={"hr": "estimate"})
            tmp["effect_type"] = "HR"
            frames.append(tmp)

    if not utilization_results.empty and {"term", "or", "ci_low", "ci_high", "model"}.issubset(utilization_results.columns):
        tmp = utilization_results[["term", "or", "ci_low", "ci_high", "model"]].copy()
        tmp = tmp.rename(columns={"or": "estimate"})
        tmp["effect_type"] = "OR"
        frames.append(tmp)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def run_all_analyses(
    analytic_df: pd.DataFrame,
    post_index_bmi_df: pd.DataFrame,
    config: dict,
) -> AnalysisBundle:
    threshold = int(config["small_cell_threshold"])
    notes: list[str] = []
    model_store: dict[str, object] = {}
    logit_diag_chunks: list[pd.DataFrame] = []

    df = prepare_analysis_df(analytic_df, config, notes=notes)

    table1 = build_table1(df)
    severity = build_severity_baseline_table(df)
    bmi_missing = build_bmi_missingness_table(df, threshold=threshold, notes=notes)
    complete_vs_missing, ipw_bmi_df = bmi_missingness_selection_models(
        df,
        config=config,
        threshold=threshold,
        notes=notes,
        model_store=model_store,
        diagnostics_store=logit_diag_chunks,
    )

    logistic_main, logistic_interaction, interaction_results = run_logistic_models(
        df,
        config=config,
        threshold=threshold,
        notes=notes,
        model_store=model_store,
        diagnostics_store=logit_diag_chunks,
    )
    extra_interaction = run_additional_interaction_models(
        df=df,
        config=config,
        threshold=threshold,
        notes=notes,
        model_store=model_store,
        diagnostics_store=logit_diag_chunks,
    )
    interaction_results = pd.concat([interaction_results, extra_interaction], ignore_index=True, sort=False)

    cox_results, ph_results, cox_diagnostics, cox_model_df = run_cox_models(
        df,
        config=config,
        threshold=threshold,
        notes=notes,
        model_store=model_store,
    )
    if not ph_results.empty:
        interaction_results = pd.concat([interaction_results, ph_results], ignore_index=True, sort=False)

    ps_results, balance = build_ps_and_outcomes(
        df,
        config=config,
        threshold=threshold,
        notes=notes,
        model_store=model_store,
    )
    notes.append(
        "Primary causal estimates use MI-pooled weighted/doubly-robust PS models; pattern-mixture deltas remain sensitivity analyses."
    )

    util_results = utilization_bias_analysis(
        df,
        config=config,
        threshold=threshold,
        notes=notes,
        model_store=model_store,
        diagnostics_store=logit_diag_chunks,
    )
    weight_change = weight_change_analysis(
        baseline_df=df,
        post_index_bmi_df=post_index_bmi_df,
        config=config,
        threshold=threshold,
        notes=notes,
        model_store=model_store,
    )

    km_curve_data = build_km_curve_data(df, config)
    forest_ready = build_forest_ready(
        logistic_main=logistic_main,
        cox_results=cox_results,
        ps_results=ps_results,
        utilization_results=util_results,
    )

    suppression_exclusions = pd.DataFrame(columns=["file", "policy_note", "suppression_columns"])
    logit_diagnostics = (
        pd.concat(logit_diag_chunks, ignore_index=True, sort=False)
        if logit_diag_chunks
        else pd.DataFrame(columns=["analysis", "scope", "level", "n", "events", "nonevents", "event_rate"])
    )

    artifacts: dict[str, object] = {
        "datasets": {
            "raw_analytic_df": analytic_df,
            "analysis_df": df,
            "cox_model_df": cox_model_df,
            "post_index_bmi_df": post_index_bmi_df,
            "bmi_ipw_df": ipw_bmi_df,
        },
        "models": model_store,
        "diagnostics": {
            "logit_diagnostics": logit_diagnostics,
            "cox_diagnostics": cox_diagnostics,
            "balance_diagnostics": balance,
        },
    }

    return AnalysisBundle(
        table1=table1,
        severity_table=severity,
        bmi_missingness=bmi_missing,
        complete_vs_missing=complete_vs_missing,
        logistic_main=logistic_main,
        logistic_interaction=logistic_interaction,
        interaction_results=interaction_results,
        cox_results=cox_results,
        ps_results=ps_results,
        balance_diagnostics=balance,
        utilization_results=util_results,
        weight_change_results=weight_change,
        forest_ready=forest_ready,
        km_curve_data=km_curve_data,
        suppression_exclusions=suppression_exclusions,
        logit_diagnostics=logit_diagnostics,
        cox_diagnostics=cox_diagnostics,
        notes=notes,
        artifacts=artifacts,
    )
