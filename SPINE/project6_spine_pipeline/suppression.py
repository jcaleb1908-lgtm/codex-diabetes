"""All of Us small-cell suppression helpers."""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

POLICY_NOTE = "Excluded due to All of Us cell-size policy (n<20)."


def _infer_count_columns(df: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    tokens = ("n", "count", "events", "total", "denominator", "numerator", "person_time")
    for col in df.columns:
        lower = col.lower()
        if any(tok in lower for tok in tokens) and pd.api.types.is_numeric_dtype(df[col]):
            candidates.append(col)
    return candidates


def suppress_small_cells(
    df: pd.DataFrame,
    threshold: int = 20,
    count_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept_rows, excluded_rows) where excluded rows violate any count threshold."""
    if df.empty:
        return df.copy(), df.copy()

    cols = list(count_columns) if count_columns is not None else _infer_count_columns(df)
    cols = [c for c in cols if c in df.columns]

    if not cols:
        return df.copy(), pd.DataFrame(columns=[*df.columns, "policy_note", "suppression_columns"])

    mask = pd.Series(False, index=df.index)
    for col in cols:
        mask = mask | (df[col].fillna(0) < threshold)

    kept = df.loc[~mask].copy()
    excluded = df.loc[mask].copy()
    if not excluded.empty:
        excluded["policy_note"] = POLICY_NOTE
        excluded["suppression_columns"] = ",".join(cols)
        logging.warning(
            "Suppressed %s rows due to n<%s policy. columns=%s",
            len(excluded),
            threshold,
            cols,
        )
    return kept, excluded


def model_is_policy_compliant(
    n: int,
    events: int,
    threshold: int,
    *,
    require_nonevents: bool = True,
) -> tuple[bool, str]:
    nonevents = n - events
    if n < threshold:
        return False, POLICY_NOTE
    if events < threshold:
        return False, POLICY_NOTE
    if require_nonevents and nonevents < threshold:
        return False, POLICY_NOTE
    return True, ""


def append_policy_note(df: pd.DataFrame, note: str = POLICY_NOTE) -> pd.DataFrame:
    out = df.copy()
    out["policy_note"] = note
    return out
