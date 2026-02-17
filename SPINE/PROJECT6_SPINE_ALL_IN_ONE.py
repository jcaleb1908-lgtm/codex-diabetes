#!/usr/bin/env python3
"""PROJECT 6 all-in-one launcher for All of Us Controlled Tier.

Canonical analysis logic lives in ``project6_spine_pipeline``.
This wrapper preserves single-file execution ergonomics while routing to
that maintained implementation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from project6_spine_pipeline.main import PipelineRunResult, main


def display_output_tables(output_dir: Path) -> None:
    """Print all CSV outputs in a notebook/script-friendly format."""
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)

    for csv_path in sorted(output_dir.glob("*.csv")):
        print(f"\n## {csv_path.name}")
        df = pd.read_csv(csv_path)
        if df.empty:
            print("[empty]")
        else:
            print(df.to_string(index=False))


def run_and_display() -> PipelineRunResult:
    result = main()
    display_output_tables(result.output_dir)
    return result


if __name__ == "__main__":
    run_and_display()
