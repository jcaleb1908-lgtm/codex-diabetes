"""Main entrypoint for Project 6 spine pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import pandas as pd

from .analysis import AnalysisBundle, run_all_analyses
from .bq_utils import create_bq_client, validate_dataset_id
from .cohort import CohortData, fetch_cohort_data
from .config import ASSUMPTIONS, CHANGE_LOG, CONFIG, CONCEPTS, REQUIRED_OUTPUT_FILES, ensure_output_dir, validate_config
from .reporting import write_report
from .suppression import suppress_small_cells


@dataclass
class PipelineRunResult:
    output_dir: Path
    generated_files: list[str]
    cohort_data: CohortData
    analyses: AnalysisBundle
    notes: list[str]
    suppression_log: pd.DataFrame


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _print_df(label: str, df: pd.DataFrame, max_rows: int = 30) -> None:
    print(f"\n===== {label} =====")
    if df.empty:
        print("[empty]")
        return
    if len(df) > max_rows:
        print(df.head(max_rows).to_string(index=False))
        print(f"... ({len(df)} rows total)")
    else:
        print(df.to_string(index=False))


def _save_with_policy(
    *,
    file_name: str,
    df: pd.DataFrame,
    output_dir: Path,
    threshold: int,
    suppression_rows: list[pd.DataFrame],
    count_columns: list[str] | None = None,
    print_tables: bool = False,
    print_max_rows: int = 30,
) -> Path:
    kept, excluded = suppress_small_cells(df, threshold=threshold, count_columns=count_columns)
    out_path = output_dir / file_name
    kept.to_csv(out_path, index=False)

    if not excluded.empty:
        tmp = excluded.copy()
        tmp["file"] = file_name
        suppression_rows.append(tmp)

    logging.info("Saved %s (%s rows)", file_name, len(kept))
    if logging.getLogger().isEnabledFor(logging.DEBUG) and not kept.empty:
        logging.debug("%s preview:\n%s", file_name, kept.head(20).to_string(index=False))
    if print_tables:
        _print_df(file_name, kept, max_rows=print_max_rows)
    return out_path


def _verify_outputs(output_dir: Path, notes: list[str]) -> None:
    for file_name in REQUIRED_OUTPUT_FILES:
        if not (output_dir / file_name).exists():
            notes.append(f"Missing expected output artifact: {file_name}")


def main() -> PipelineRunResult:
    _configure_logging()
    validate_config()

    output_dir = ensure_output_dir()
    dataset = validate_dataset_id(CONFIG["dataset"])
    threshold = int(CONFIG["small_cell_threshold"])
    print_tables = bool(CONFIG.get("print_tables_in_notebook", False))
    print_max_rows = int(CONFIG.get("print_table_max_rows", 30))

    logging.info("Starting Project 6 pipeline. dataset=%s", dataset)
    logging.info("Using dataset: %s", dataset)
    logging.info("Output directory: %s", output_dir)

    client = create_bq_client(location=CONFIG["bq_location"])

    cohort_data = fetch_cohort_data(client, dataset, CONFIG, CONCEPTS)

    suppression_rows: list[pd.DataFrame] = []
    generated_files: list[str] = []

    cohort_flow_path = _save_with_policy(
        file_name="cohort_flow.csv",
        df=cohort_data.cohort_flow,
        output_dir=output_dir,
        threshold=threshold,
        suppression_rows=suppression_rows,
        count_columns=["n"],
        print_tables=print_tables,
        print_max_rows=print_max_rows,
    )
    generated_files.append(str(cohort_flow_path.name))

    analyses = run_all_analyses(
        analytic_df=cohort_data.analytic_df,
        post_index_bmi_df=cohort_data.post_index_bmi_df,
        config=CONFIG,
    )

    output_map: list[tuple[str, pd.DataFrame, list[str] | None]] = [
        ("table1_baseline_by_treatment.csv", analyses.table1, ["n"]),
        ("severity_baseline_table.csv", analyses.severity_table, ["n"]),
        (
            "bmi_missingness_table.csv",
            analyses.bmi_missingness,
            ["n_total", "n_bmi_missing", "n_bmi_present"],
        ),
        ("complete_vs_missing_bmi_risk.csv", analyses.complete_vs_missing, ["n", "events"]),
        ("logistic_main_hc3.csv", analyses.logistic_main, ["n_total", "events"]),
        ("logistic_interaction_obesity.csv", analyses.logistic_interaction, ["n_total", "events"]),
        ("interaction_results.csv", analyses.interaction_results, ["n_total", "events"]),
        ("logit_model_diagnostics.csv", analyses.logit_diagnostics, ["n", "events", "nonevents"]),
        ("cox_model_diagnostics.csv", analyses.cox_diagnostics, ["n", "events", "nonevents"]),
        ("cox_time_to_spine.csv", analyses.cox_results, ["n_total", "events", "person_time_days"]),
        (
            "propensity_score_results.csv",
            analyses.ps_results,
            ["n_total"],
        ),
        ("balance_diagnostics.csv", analyses.balance_diagnostics, ["n_treated", "n_control"]),
        ("utilization_bias_results.csv", analyses.utilization_results, ["n", "events", "n_total"]),
        ("weight_change_analysis.csv", analyses.weight_change_results, ["n", "events", "n_total"]),
        ("forest_plot_ready.csv", analyses.forest_ready, None),
        ("km_curve_data.csv", analyses.km_curve_data, ["n_at_risk", "n_events", "n_censored"]),
    ]

    for file_name, df, count_cols in output_map:
        path = _save_with_policy(
            file_name=file_name,
            df=df,
            output_dir=output_dir,
            threshold=threshold,
            suppression_rows=suppression_rows,
            count_columns=count_cols,
            print_tables=print_tables,
            print_max_rows=print_max_rows,
        )
        generated_files.append(path.name)

    suppression_log = (
        pd.concat(suppression_rows, ignore_index=True, sort=False)
        if suppression_rows
        else pd.DataFrame(columns=["file", "policy_note", "suppression_columns"])
    )

    if suppression_log.empty:
        logging.info("No rows suppressed under AoU policy threshold n<%s", threshold)
    else:
        logging.warning("Suppression applied to %s rows total.", len(suppression_log))

    notes = list(analyses.notes)
    if CONFIG.get("temp_dataset"):
        notes.append(f"Intermediate cache tables were materialized in `{CONFIG['temp_dataset']}`.")

    _verify_outputs(output_dir, notes)

    report_path = write_report(
        output_dir=output_dir,
        change_log=CHANGE_LOG,
        assumptions=ASSUMPTIONS,
        cohort_flow=cohort_data.cohort_flow,
        generated_files=generated_files,
        notes=notes,
        suppression_log=suppression_log[[c for c in ["file", "policy_note", "suppression_columns"] if c in suppression_log.columns]],
    )
    generated_files.append(report_path.name)

    logging.info("Pipeline complete. Generated files:")
    for fp in sorted(generated_files):
        logging.info("- %s", fp)

    return PipelineRunResult(
        output_dir=output_dir,
        generated_files=sorted(generated_files),
        cohort_data=cohort_data,
        analyses=analyses,
        notes=notes,
        suppression_log=suppression_log,
    )


if __name__ == "__main__":
    main()
