"""Cleaning helpers for defect prediction datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd

TEXT_COLUMNS = ["module_id", "project_name", "commit_text"]


def clean_dataset(
    df: pd.DataFrame,
    deduplicate_by_module_id: bool = False,
    return_summary: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Apply baseline cleaning rules to a dataset."""
    cleaned = df.copy()
    summary: dict[str, Any] = {
        "rows_before": len(cleaned),
        "exact_duplicates_removed": 0,
        "module_duplicates_removed": 0,
        "rows_missing_label_removed": 0,
        "numeric_columns_imputed": [],
    }

    for col in TEXT_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna("").astype(str).str.strip()

    before_drop_duplicates = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    summary["exact_duplicates_removed"] = before_drop_duplicates - len(cleaned)

    if deduplicate_by_module_id and "module_id" in cleaned.columns:
        before_module_dedup = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=["module_id"], keep="first")
        summary["module_duplicates_removed"] = before_module_dedup - len(cleaned)

    if "label" in cleaned.columns:
        before_dropna_label = len(cleaned)
        cleaned = cleaned[cleaned["label"].notna()].copy()
        summary["rows_missing_label_removed"] = before_dropna_label - len(cleaned)
        if not cleaned.empty:
            cleaned["label"] = cleaned["label"].astype(int)

    numeric_cols = [col for col in cleaned.select_dtypes(include=["number"]).columns if col != "label"]
    for col in numeric_cols:
        if cleaned[col].isna().any():
            median_value = cleaned[col].median()
            if pd.isna(median_value):
                cleaned[col] = cleaned[col].fillna(0)
            else:
                cleaned[col] = cleaned[col].fillna(median_value)
            summary["numeric_columns_imputed"].append(col)

    summary["rows_after"] = len(cleaned)
    return (cleaned, summary) if return_summary else cleaned
