"""Schema harmonization utilities for multi-source datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# GHPR text columns used to compose commit_text. We deliberately omit
# DIFF_CODE (source diffs leak filename/identifier tokens that match
# the bug fix pair) and PROJECT_DESCRIPTION/PROJECT_LABEL (per-project
# constants behave like project IDs). The diff content is preserved
# separately by the GHPR adapter via a dedicated diff_text column.
GHPR_TEXT_COLUMNS = ["COMMIT_DESCRIPTION", "PR_TITLE", "PR_DESCRIPTION"]

COLUMN_ALIASES = {
    "label": ["label", "class", "bug", "bugs", "defect", "defects", "is_buggy", "is_defective"],
    "module_id": ["module_id", "name", "module", "module_name", "file", "filename"],
    "project_name": ["project_name", "project", "repository", "repo"],
    "commit_text": [
        "commit_text",
        "commit_message",
        "commit_msg",
        "message",
        "log",
        "commit",
        "COMMIT_DESCRIPTION",
        "PR_TITLE",
        "PR_DESCRIPTION",
    ],
}

STRING_LABEL_MAP = {
    "true": 1,
    "false": 0,
    "yes": 1,
    "no": 0,
    "buggy": 1,
    "clean": 0,
    "defective": 1,
    "non-defective": 0,
    "defect": 1,
    "nondefective": 0,
}

# Canonical project names for the McCabe metrics actually consumed by the
# baseline feature pipeline. Some sources (e.g. pc1 from "Promise + BPD")
# expose iv(G) with an upper-case G; the pipeline expects the lower-case
# variant. We deliberately do NOT lower-case unrelated Halstead columns so
# tests and ad-hoc analyses that rely on their original casing keep working.
CANONICAL_MCCABE_METRICS = {
    "loc": "loc",
    "v(g)": "v(g)",
    "ev(g)": "ev(g)",
    "iv(g)": "iv(g)",
    "branchcount": "branchCount",
}

# Datasets that historically lack commit text. Keeping an empty placeholder
# column was confusing for downstream feature builders, so the unified frame
# explicitly drops the empty column for these datasets.
PROMISE_DATASETS_WITHOUT_TEXT = {"cm1", "jm1", "kc1", "pc1"}

def _canonicalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for col in df.columns:
        canonical = CANONICAL_MCCABE_METRICS.get(col.lower())
        if canonical is not None and col != canonical:
            rename_map[col] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _find_matching_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered_map:
            return lowered_map[candidate.lower()]
    return None


def _rename_to_standard_schema(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    rename_map: dict[str, str] = {}
    for standard_name, aliases in COLUMN_ALIASES.items():
        matched = _find_matching_column(renamed, aliases)
        if matched and matched != standard_name:
            rename_map[matched] = standard_name
    return renamed.rename(columns=rename_map)


def _normalize_label_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        values = numeric.astype(float)
        non_null = values.dropna()
        is_integer_valued = bool(((non_null % 1) == 0).all())
        if is_integer_valued:
            unique_values = set(non_null.astype(int).unique().tolist())
            if unique_values.issubset({0, 1}):
                return values.astype(int)
        if (values >= 0).all():
            return (values > 0).astype(int)
        raise ValueError(f"Could not normalize label values: {sorted(set(non_null.tolist()))[:10]}")

    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map(STRING_LABEL_MAP)
    unresolved = normalized[mapped.isna()].dropna().unique().tolist()
    if unresolved:
        raise ValueError(f"Could not normalize label values: {unresolved[:10]}")
    return mapped.astype(int)


def _compose_commit_text(df: pd.DataFrame) -> pd.Series:
    text_frame = df.copy()
    for col in GHPR_TEXT_COLUMNS:
        if col in text_frame.columns:
            text_frame[col] = text_frame[col].fillna("").astype(str).str.strip()
        else:
            text_frame[col] = ""

    return text_frame[GHPR_TEXT_COLUMNS].apply(
        lambda row: " ".join([value for value in row.tolist() if value]).strip(),
        axis=1,
    )


def unify_schema(
    df: pd.DataFrame,
    dataset_name: str | None = None,
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize dataset columns to the project standard schema."""
    unified = _rename_to_standard_schema(df)
    unified = _canonicalize_metric_columns(unified)

    if column_map:
        unified = unified.rename(columns=column_map)

    if "module_id" not in unified.columns:
        unified["module_id"] = [f"row_{idx}" for idx in range(len(unified))]

    if "project_name" not in unified.columns:
        fallback_project = Path(dataset_name).stem if dataset_name else "unknown_project"
        unified["project_name"] = fallback_project

    if "commit_text" not in unified.columns:
        unified["commit_text"] = ""
    else:
        unified["commit_text"] = unified["commit_text"].fillna("").astype(str)

    if dataset_name and "ghpr" in dataset_name.lower():
        composed_text = _compose_commit_text(unified)
        unified["commit_text"] = unified["commit_text"].where(unified["commit_text"].str.strip().ne(""), composed_text)

    if "label" not in unified.columns and "class" in unified.columns:
        unified["label"] = unified["class"]

    if "label" in unified.columns:
        unified["label"] = _normalize_label_series(unified["label"])

    # Use the stem so callers can pass either "pc1" or "pc1.csv" as dataset_name.
    dataset_stem = Path(dataset_name).stem.strip().lower() if dataset_name else ""
    if (
        dataset_stem
        and dataset_stem in PROMISE_DATASETS_WITHOUT_TEXT
        and "commit_text" in unified.columns
    ):
        text_series = unified["commit_text"].fillna("").astype(str).str.strip()
        if not text_series.ne("").any():
            unified = unified.drop(columns=["commit_text"])

    return unified
