"""Small value coercion helpers shared across scripts and services."""

from __future__ import annotations

from typing import Any

import pandas as pd


def coerce_bool(value: Any, default: bool = False) -> bool:
    """Coerce common Python/CSV truthy and falsy values safely."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n", ""}:
            return False
    return default


def coerce_float(value: Any, default: float | None = None) -> float | None:
    """Coerce a value to float, treating null-like values as default."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
