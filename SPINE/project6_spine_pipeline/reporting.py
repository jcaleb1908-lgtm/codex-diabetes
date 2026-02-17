"""Report generation utilities for Project 6 outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _fmt_pct(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{100 * float(x):.1f}%"


def write_report(
    *,
    output_dir: Path,
    change_log: list[str],
    assumptions: list[str],
    cohort_flow: pd.DataFrame,
    generated_files: list[str],
    notes: list[str],
    suppression_log: pd.DataFrame,
) -> Path:
    report_path = output_dir / "REPORT.md"

    lines: list[str] = []
    lines.append("# PROJECT 6 REPORT: Degenerative Spine Disease and Diabetes Medications")
    lines.append("")

    lines.append("## Change Log")
    for entry in change_log:
        lines.append(f"- {entry}")
    lines.append("")

    lines.append("## Assumptions")
    for entry in assumptions:
        lines.append(f"- {entry}")
    lines.append("")

    lines.append("## Cohort Flow")
    if cohort_flow.empty:
        lines.append("- Cohort flow unavailable.")
    else:
        for _, row in cohort_flow.iterrows():
            step = row.get("step", "step")
            n = row.get("n", "NA")
            lines.append(f"- {step}: {n}")
    lines.append("")

    lines.append("## Generated Artifacts")
    for fp in sorted(generated_files):
        lines.append(f"- `{fp}`")
    lines.append("")

    lines.append("## Policy Exclusions (AoU n<20)")
    if suppression_log.empty:
        lines.append("- No table rows were removed by suppression checks.")
    else:
        for _, row in suppression_log.iterrows():
            file_name = row.get("file", "unknown")
            policy_note = row.get("policy_note", "")
            lines.append(f"- `{file_name}`: {policy_note}")
    lines.append("")

    lines.append("## Notes")
    if not notes:
        lines.append("- None.")
    else:
        for note in notes:
            lines.append(f"- {note}")
    lines.append("")

    lines.append("## Interpretation Guardrails")
    lines.append("- Binary 2-year models are secondary/replication analyses.")
    lines.append("- 2-year-censored time-to-event models include participants with positive observed follow-up and treat shorter follow-up as censoring.")
    lines.append("- Outputs excluded under AoU small-cell policy are not interpreted.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
