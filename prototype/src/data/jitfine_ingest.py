"""Ingest helpers for the JIT-Fine cross-project dataset.

JIT-Fine bundles 21 Apache Java projects in a single train/valid/test set with
native pickle artifacts that already contain commit messages, change metrics,
and JIT defect labels. This module loads the three feature pickles and projects
them into the project-standard schema (`module_id`, `label`, `commit_text`,
`project_name`, plus the JIT-Fine change metric columns).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.logging import get_logger
from src.utils.paths import RAW_DATA_DIR

JITFINE_RAW_DIR_NAME = "JITFine"
JITFINE_DATASET_NAME = "jitfine"

logger = get_logger(__name__)

JITFINE_FEATURE_FILES: dict[str, str] = {
    "train": "features_train.pkl",
    "val": "features_valid.pkl",
    "test": "features_test.pkl",
}

JITFINE_METRIC_COLUMNS: tuple[str, ...] = (
    "la",
    "ld",
    "nf",
    "ns",
    "nd",
    "entropy",
    "ndev",
    "lt",
    "nuc",
    "age",
    "exp",
    "rexp",
    "sexp",
)

JITFINE_RAW_FEATURE_COLUMNS: tuple[str, ...] = (
    "commit_hash",
    "project",
    "commit_message",
    "is_buggy_commit",
    *JITFINE_METRIC_COLUMNS,
)


def jitfine_raw_dir(raw_dir: Path | str | None = None) -> Path:
    """Return the JIT-Fine raw subdirectory under the project raw data root."""
    base = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    return base / JITFINE_RAW_DIR_NAME


def resolve_jitfine_feature_paths(raw_dir: Path | str | None = None) -> dict[str, Path]:
    """Resolve the canonical layout for the three JIT-Fine feature pickles."""
    base = jitfine_raw_dir(raw_dir)
    return {split: base / filename for split, filename in JITFINE_FEATURE_FILES.items()}


def jitfine_artifacts_available(raw_dir: Path | str | None = None) -> bool:
    """Return True only when all three JIT-Fine feature pickles are present."""
    return all(path.exists() for path in resolve_jitfine_feature_paths(raw_dir).values())


def _load_feature_pickle(path: Path) -> pd.DataFrame:
    """Load one JIT-Fine feature pickle into a DataFrame."""
    df = pd.read_pickle(path)
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"JIT-Fine feature pickle {path} is not a DataFrame.")
    missing = [column for column in JITFINE_RAW_FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"JIT-Fine feature pickle {path} is missing columns: {missing}")
    return df


def _coerce_metric_columns(df: pd.DataFrame, metric_columns: Iterable[str]) -> pd.DataFrame:
    """Force metric columns to numeric float64; non-numeric values become NaN."""
    coerced = df.copy()
    for column in metric_columns:
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce").astype("float64")
    return coerced


def build_jitfine_dataset(
    raw_dir: Path | str | None = None,
    metric_columns: Iterable[str] = JITFINE_METRIC_COLUMNS,
) -> pd.DataFrame:
    """Build the unified JIT-Fine dataset across 21 Apache Java projects."""
    paths = resolve_jitfine_feature_paths(raw_dir)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing JIT-Fine feature pickle(s): {missing}")

    selected_metric_columns = [column for column in metric_columns if column in JITFINE_METRIC_COLUMNS]
    frames: list[pd.DataFrame] = []
    for split_name, split_path in paths.items():
        raw_df = _load_feature_pickle(split_path)
        keep_columns = [
            "commit_hash",
            "project",
            "commit_message",
            "is_buggy_commit",
            *selected_metric_columns,
        ]
        frame = raw_df.loc[:, keep_columns].copy()
        frame = _coerce_metric_columns(frame, selected_metric_columns)
        frame["jitline_split"] = split_name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.rename(
        columns={
            "commit_hash": "module_id",
            "project": "project_name",
            "commit_message": "commit_text",
            "is_buggy_commit": "label",
        }
    )
    combined["module_id"] = combined["module_id"].astype(str).str.strip()
    combined["project_name"] = combined["project_name"].astype(str)
    combined["commit_text"] = combined["commit_text"].fillna("").astype(str)
    combined["label"] = pd.to_numeric(combined["label"], errors="coerce")
    missing_label_count = int(combined["label"].isna().sum())
    if missing_label_count:
        logger.warning(
            "JIT-Fine ingest dropped %s row(s) with missing/non-numeric labels",
            missing_label_count,
        )
        combined = combined.loc[combined["label"].notna()].copy()
    combined["label"] = combined["label"].astype(int)
    combined["dataset_name"] = JITFINE_DATASET_NAME

    duplicated = combined["module_id"].duplicated()
    if duplicated.any():
        combined = combined.loc[~duplicated].reset_index(drop=True)

    return combined


__all__ = [
    "JITFINE_DATASET_NAME",
    "JITFINE_FEATURE_FILES",
    "JITFINE_METRIC_COLUMNS",
    "JITFINE_RAW_DIR_NAME",
    "JITFINE_RAW_FEATURE_COLUMNS",
    "build_jitfine_dataset",
    "jitfine_artifacts_available",
    "jitfine_raw_dir",
    "resolve_jitfine_feature_paths",
]