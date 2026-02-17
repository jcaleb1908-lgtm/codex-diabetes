"""BigQuery helper functions for parameterized cohort queries."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

try:
    from google.api_core.exceptions import BadRequest, Forbidden, GoogleAPICallError, NotFound
    from google.cloud import bigquery
except Exception:  # pragma: no cover - local environments may not include google sdk
    BadRequest = Forbidden = GoogleAPICallError = NotFound = Exception
    bigquery = None  # type: ignore[assignment]

DATASET_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass
class QueryArtifact:
    name: str
    row_count: int
    table_fqn: str | None = None


def validate_dataset_id(dataset: str) -> str:
    if not dataset:
        raise ValueError("WORKSPACE_CDR is empty.")
    if not DATASET_PATTERN.match(dataset):
        raise ValueError(
            "Invalid dataset id. Allowed characters: letters, numbers, underscore, dot, hyphen."
        )
    return dataset


def create_bq_client(location: str = "US") -> bigquery.Client:
    if bigquery is None:
        raise ImportError(
            "google-cloud-bigquery is not installed. Run inside AoU Workbench or install: "
            "`pip install google-cloud-bigquery`."
        )
    return bigquery.Client(location=location)


def scalar_param(name: str, ptype: str, value: object) -> bigquery.ScalarQueryParameter:
    if bigquery is None:
        raise ImportError("google-cloud-bigquery is required for query parameters.")
    return bigquery.ScalarQueryParameter(name, ptype, value)


def array_param(name: str, ptype: str, values: Sequence[object]) -> bigquery.ArrayQueryParameter:
    if bigquery is None:
        raise ImportError("google-cloud-bigquery is required for query parameters.")
    return bigquery.ArrayQueryParameter(name, ptype, list(values))


def run_query(
    client: bigquery.Client,
    sql: str,
    params: Iterable[bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter] | None,
    *,
    job_name: str,
) -> pd.DataFrame:
    query_parameters = list(params) if params else []
    job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
    logging.info("Running query: %s", job_name)
    try:
        job = client.query(sql, job_config=job_config)
        result = job.result()
        df = result.to_dataframe(create_bqstorage_client=True)
        logging.info("Finished query: %s | rows=%s", job_name, len(df))
        return df
    except (BadRequest, Forbidden, NotFound, GoogleAPICallError) as exc:
        logging.exception("BigQuery query failed: %s", job_name)
        raise RuntimeError(f"BigQuery query failed ({job_name}): {exc}") from exc


def materialize_query(
    client: bigquery.Client,
    select_sql: str,
    params: Iterable[bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter] | None,
    *,
    table_fqn: str,
    job_name: str,
) -> QueryArtifact:
    query_parameters = list(params) if params else []
    sql = f"CREATE OR REPLACE TABLE `{table_fqn}` AS\n{select_sql}"
    job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
    logging.info("Materializing query: %s -> %s", job_name, table_fqn)
    try:
        job = client.query(sql, job_config=job_config)
        job.result()
        cnt_df = client.query(f"SELECT COUNT(*) AS n FROM `{table_fqn}`").result().to_dataframe()
        row_count = int(cnt_df.loc[0, "n"]) if not cnt_df.empty else 0
        return QueryArtifact(name=job_name, row_count=row_count, table_fqn=table_fqn)
    except (BadRequest, Forbidden, NotFound, GoogleAPICallError) as exc:
        logging.exception("BigQuery materialization failed: %s", job_name)
        raise RuntimeError(f"BigQuery materialization failed ({job_name}): {exc}") from exc
