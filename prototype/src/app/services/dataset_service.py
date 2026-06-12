"""Dataset access helpers for the MVC backend."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.app.state import StatusMessage
from src.features.metrics_features import get_default_metric_registry
from src.utils.io import read_parquet
from src.utils.paths import PROCESSED_DATA_DIR

BASELINE_DATASETS = ["cm1", "jm1", "kc1", "pc1"]
DEFAULT_METRIC_REGISTRY = get_default_metric_registry()
DEFAULT_METRICS = DEFAULT_METRIC_REGISTRY["core"]
PAPER_METRICS = DEFAULT_METRIC_REGISTRY["paper_extended"]
COMMIT_TEXT_CANDIDATES = ["commit_text", "commit_message", "commit_msg", "message", "log", "commit"]


def dataset_processed_path(dataset_name: str) -> Path:
    """Return the processed parquet path for one dataset."""
    return PROCESSED_DATA_DIR / f"{dataset_name}_clean.parquet"


def load_processed_dataset(dataset_name: str) -> pd.DataFrame:
    """Load a processed dataset when available."""
    path = dataset_processed_path(dataset_name)
    if not path.exists():
        return pd.DataFrame()
    return read_parquet(path)


def select_metric_columns(df: pd.DataFrame, preferred_metrics: list[str] | None = None) -> list[str]:
    """Return the configured metrics that exist in the processed dataset."""
    target_metrics = preferred_metrics or PAPER_METRICS
    return [column for column in target_metrics if column in df.columns]


def select_commit_text_column(df: pd.DataFrame) -> str | None:
    """Return the best available commit text column, if any."""
    for column in COMMIT_TEXT_CANDIDATES:
        if column in df.columns:
            return column
    return None


def describe_dataset_availability(dataset_name: str) -> StatusMessage:
    """Return availability metadata for one processed dataset."""
    path = dataset_processed_path(dataset_name)
    if not path.exists():
        return StatusMessage(
            available=False,
            message="Processed dataset not found.",
            details={"dataset_name": dataset_name, "path": str(path)},
        )

    return StatusMessage(
        available=True,
        message="Processed dataset is available.",
        details={"dataset_name": dataset_name, "path": str(path)},
    )


def build_sample_rows(
    dataset_name: str, limit: int = 10
) -> tuple[list[dict], list[str], pd.DataFrame, StatusMessage]:
    """Build sample module rows for UI display and prediction."""
    dataset_status = describe_dataset_availability(dataset_name)
    if not dataset_status.available:
        return [], [], pd.DataFrame(), dataset_status

    try:
        df = load_processed_dataset(dataset_name)
    except Exception as exc:
        return [], [], pd.DataFrame(), StatusMessage(
            available=False,
            message="Failed to load processed dataset.",
            details={"dataset_name": dataset_name, "error": str(exc)},
        )

    if df.empty:
        return [], [], pd.DataFrame(), StatusMessage(
            available=False,
            message="Processed dataset is empty.",
            details={"dataset_name": dataset_name},
        )

    metric_columns = select_metric_columns(df)
    commit_text_column = select_commit_text_column(df)
    display_columns = [column for column in ["module_id", *metric_columns, commit_text_column, "label"] if column and column in df.columns]
    sample_df = df[display_columns].head(limit).copy().reset_index(drop=True)
    dataset_status.details.update(
        {
            "row_count": int(len(df)),
            "sample_row_count": int(len(sample_df)),
            "metric_columns": metric_columns,
            "paper_metric_columns": metric_columns,
            "commit_text_column": commit_text_column,
            "has_commit_text": commit_text_column is not None,
            "commit_text_available": commit_text_column is not None,
        }
    )
    return sample_df.to_dict("records"), metric_columns, sample_df, dataset_status


__all__ = [
    "BASELINE_DATASETS",
    "COMMIT_TEXT_CANDIDATES",
    "DEFAULT_METRIC_REGISTRY",
    "DEFAULT_METRICS",
    "PAPER_METRICS",
    "build_sample_rows",
    "dataset_processed_path",
    "describe_dataset_availability",
    "load_processed_dataset",
    "select_commit_text_column",
    "select_metric_columns",
]
