"""Ingest helpers for the JITLine + DeepJIT replication datasets.

The replication package on Zenodo bundles two complementary artifacts per
project (`openstack` and `qt`):

* JITLine change metrics CSV (`<project>_metrics.csv`) holding ApacheJIT-style
  change metrics keyed by `commit_id`.
* DeepJIT pickle bundles (`<project>_train.pkl`, `<project>_test.pkl`) holding
  commit identifiers, binary defect labels, and tokenized commit messages.

This module joins the two sources into a unified per-commit table that fits
the project's metrics + commit-text schema (`module_id`, `label`,
`commit_text`, `project_name`, plus the JITLine metric columns).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.logging import get_logger
from src.utils.paths import RAW_DATA_DIR

logger = get_logger(__name__)

JITLINE_PROJECTS: tuple[str, ...] = ("openstack", "qt")
JITLINE_METRIC_COLUMNS: tuple[str, ...] = (
    "la",
    "ld",
    "nf",
    "nd",
    "ns",
    "ent",
    "nrev",
    "rtime",
    "hcmt",
    "self",
    "ndev",
    "age",
    "nuc",
    "app",
    "aexp",
    "rexp",
    "arexp",
    "rrexp",
    "asexp",
    "rsexp",
    "asawr",
    "rsawr",
)
DROPPED_LEAKY_METRICS: tuple[str, ...] = ("bugcount", "fixcount", "revd", "tcmt", "oexp", "orexp", "osexp", "osawr")
JITLINE_REQUIRED_TEXT_FIELDS: tuple[str, ...] = ("commit_id", "label", "commit_text")
JITLINE_RAW_DIR_NAME = "JITLine"

@dataclass(frozen=True)
class JitlineSourcePaths:
    """Resolved on-disk locations for one JITLine project."""

    project: str
    metrics_csv: Path
    train_pkl: Path
    test_pkl: Path

    def all_exist(self) -> bool:
        return all(path.exists() for path in (self.metrics_csv, self.train_pkl, self.test_pkl))

    @property
    def missing_files(self) -> list[Path]:
        return [path for path in (self.metrics_csv, self.train_pkl, self.test_pkl) if not path.exists()]

def jitline_raw_dir(raw_dir: Path | str | None = None) -> Path:
    """Return the JITLine raw subdirectory under the project raw data root."""
    base = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR
    return base / JITLINE_RAW_DIR_NAME

def resolve_jitline_paths(project: str, raw_dir: Path | str | None = None) -> JitlineSourcePaths:
    """Resolve the canonical file layout for one JITLine project."""
    project_key = project.strip().lower()
    if project_key not in JITLINE_PROJECTS:
        raise ValueError(f"Unsupported JITLine project: {project!r}. Expected one of {JITLINE_PROJECTS}.")
    base = jitline_raw_dir(raw_dir)
    return JitlineSourcePaths(
        project=project_key,
        metrics_csv=base / f"{project_key}_metrics.csv",
        train_pkl=base / f"{project_key}_train.pkl",
        test_pkl=base / f"{project_key}_test.pkl",
    )

def _load_text_pickle(path: Path) -> pd.DataFrame:
    """Load one DeepJIT pickle and project the (id, label, message) triples."""
    with open(path, "rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, (list, tuple)) or len(bundle) < 3:
        raise ValueError(f"Unexpected DeepJIT pickle structure in {path}: expected (ids, labels, messages, ...)")
    ids, labels, messages = bundle[0], bundle[1], bundle[2]
    if not (len(ids) == len(labels) == len(messages)):
        raise ValueError(
            f"DeepJIT pickle {path} has misaligned arrays: ids={len(ids)}, labels={len(labels)}, messages={len(messages)}"
        )
    return pd.DataFrame(
        {
            "commit_id": [str(value).strip() for value in ids],
            "label": [int(value) for value in labels],
            "commit_text": [str(value) for value in messages],
        }
    )

def load_jitline_text_records(paths: JitlineSourcePaths) -> pd.DataFrame:
    """Concatenate the train+test DeepJIT pickles into one labeled text table."""
    train_df = _load_text_pickle(paths.train_pkl)
    train_df["jitline_split"] = "train"
    test_df = _load_text_pickle(paths.test_pkl)
    test_df["jitline_split"] = "test"
    combined = pd.concat([train_df, test_df], ignore_index=True)
    if combined["commit_id"].duplicated().any():
        combined = combined.drop_duplicates(subset=["commit_id"], keep="first").reset_index(drop=True)
    return combined

def _pick_metric_columns(metrics_df: pd.DataFrame, requested: Iterable[str]) -> list[str]:
    return [column for column in requested if column in metrics_df.columns]

def load_jitline_metrics(paths: JitlineSourcePaths, metric_columns: Iterable[str] = JITLINE_METRIC_COLUMNS) -> pd.DataFrame:
    """Load the JITLine metric CSV with `commit_id` keys and selected metric columns."""
    metrics_df = pd.read_csv(paths.metrics_csv)
    if "commit_id" not in metrics_df.columns:
        raise ValueError(f"JITLine metrics file {paths.metrics_csv} is missing the commit_id column")
    selected = ["commit_id", *_pick_metric_columns(metrics_df, metric_columns)]
    metrics_df = metrics_df.loc[:, selected].copy()
    metrics_df["commit_id"] = metrics_df["commit_id"].astype(str).str.strip()
    metrics_df = metrics_df.drop_duplicates(subset=["commit_id"], keep="first")
    return metrics_df

def build_jitline_dataset(
    project: str,
    raw_dir: Path | str | None = None,
    metric_columns: Iterable[str] = JITLINE_METRIC_COLUMNS,
) -> pd.DataFrame:
    """Build the unified per-commit dataset for one JITLine project."""
    paths = resolve_jitline_paths(project, raw_dir=raw_dir)
    if not paths.all_exist():
        missing = ", ".join(str(path) for path in paths.missing_files)
        raise FileNotFoundError(f"Missing JITLine source files for {paths.project}: {missing}")

    text_df = load_jitline_text_records(paths)
    metrics_df = load_jitline_metrics(paths, metric_columns=metric_columns)

    merged = text_df.merge(metrics_df, on="commit_id", how="left")
    metric_only_columns = [column for column in metrics_df.columns if column != "commit_id"]
    if metric_only_columns:
        unmatched = int(merged[metric_only_columns].isna().all(axis=1).sum())
        if unmatched:
            logger.warning(
                "JITLine %s: %s commit(s) have no matching metrics row (left-join NaNs)",
                paths.project,
                unmatched,
            )
    merged.insert(0, "module_id", merged["commit_id"].astype(str))
    merged.insert(1, "project_name", paths.project)
    merged["dataset_name"] = paths.project
    return merged

def available_jitline_projects(raw_dir: Path | str | None = None) -> list[str]:
    """Return the JITLine projects whose raw artifacts are all present locally."""
    base = jitline_raw_dir(raw_dir)
    if not base.exists():
        return []
    return [project for project in JITLINE_PROJECTS if resolve_jitline_paths(project, raw_dir=raw_dir).all_exist()]

__all__ = [
    "DROPPED_LEAKY_METRICS",
    "JITLINE_METRIC_COLUMNS",
    "JITLINE_PROJECTS",
    "JITLINE_RAW_DIR_NAME",
    "JITLINE_REQUIRED_TEXT_FIELDS",
    "JitlineSourcePaths",
    "available_jitline_projects",
    "build_jitline_dataset",
    "jitline_raw_dir",
    "load_jitline_metrics",
    "load_jitline_text_records",
    "resolve_jitline_paths",
]
