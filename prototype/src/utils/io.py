"""Simple IO helpers for CSV, Parquet, and JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    """Write a DataFrame to CSV, creating parent folders when needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)


def read_parquet(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    return pd.read_parquet(path, **kwargs)


def write_parquet(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    """Write a DataFrame to Parquet, creating parent folders when needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)


def write_json(data: dict[str, Any], path: str | Path, **kwargs: Any) -> None:
    """Write a JSON artifact with stable formatting."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, **kwargs)
