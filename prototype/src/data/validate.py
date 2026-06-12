"""Validation helpers for dataset schema checks."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

REQUIRED_COLUMNS = ["module_id", "label"]


def ensure_non_empty_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise an error when required columns are missing or contain no usable values."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    empty = [
        column
        for column in columns
        if df[column].isna().all() or df[column].astype(str).str.strip().eq("").all()
    ]
    if empty:
        raise ValueError(f"Columns contain no usable values: {empty}")


def validate_required_columns(df: pd.DataFrame, required_columns: list[str] | None = None) -> None:
    """Raise an error if required columns are missing."""
    required = required_columns or REQUIRED_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_dataset_schema(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
    label_column: str = "label",
) -> None:
    """Validate the basic dataset contract before downstream processing."""
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    if df.empty:
        raise ValueError("Dataset is empty")

    validate_required_columns(df, required_columns)

    if label_column in df.columns:
        label_series = df[label_column]
        if label_series.isna().any():
            raise ValueError(f"Column '{label_column}' contains missing values")

        if pd.api.types.is_bool_dtype(label_series):
            return

        numeric = pd.to_numeric(label_series, errors="coerce")
        if numeric.notna().all():
            unique_values = set(numeric.dropna().astype(int).unique().tolist())
            if not unique_values.issubset({0, 1}):
                raise ValueError(f"Column '{label_column}' must be binary after normalization")
            return

        normalized = label_series.astype(str).str.strip().str.lower()
        allowed = {"0", "1", "true", "false", "yes", "no", "buggy", "clean", "defective", "non-defective", "defect", "nondefective"}
        unresolved = [value for value in normalized.dropna().unique().tolist() if value not in allowed]
        if unresolved:
            raise ValueError(f"Column '{label_column}' contains unsupported values: {unresolved[:10]}")




def validate_text_ready_dataset(df: pd.DataFrame, text_column: str = "commit_text") -> None:
    """Validate that a dataset has usable text for hybrid training when required."""
    if text_column not in df.columns:
        raise ValueError(f"Missing required text column: {text_column}")
    if df[text_column].astype(str).str.strip().eq("").all():
        raise ValueError(f"Column '{text_column}' has no usable text")
