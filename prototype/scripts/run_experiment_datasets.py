"""Build experiment-ready datasets for the primary baseline experiments."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_project_config
from src.data.split import build_split_manifest, save_split_manifest, save_split_ids
from src.features.metrics_features import build_metrics_features, summarize_metric_coverage
from src.utils.io import read_parquet, write_csv, write_parquet
from src.utils.logging import get_logger
from src.utils.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, ensure_project_dirs
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

PRIMARY_DATASET_NAMES = ["cm1", "jm1", "kc1", "pc1"]
EXPERIMENT_DIR = PROCESSED_DATA_DIR / "experiments"
MANIFEST_PATH = INTERIM_DATA_DIR / "experiment_dataset_manifest.csv"
SUMMARY_PATH = INTERIM_DATA_DIR / "experiment_dataset_summary.csv"


def load_experiment_config() -> dict[str, Any]:
    """Load project settings used for experiment dataset construction."""
    config = load_project_config()
    return {
        "random_seed": config.get("project", {}).get("random_seed", 42),
        "test_size": config.get("split", {}).get("test_size", 0.2),
        "val_size": config.get("split", {}).get("val_size", 0.1),
        "metrics": config.get("features", {}).get("metrics", []),
    }


def discover_cleaned_datasets() -> list[Path]:
    """Return cleaned baseline datasets that are ready for experiment packaging."""
    return [PROCESSED_DATA_DIR / f"{dataset_name}_clean.parquet" for dataset_name in PRIMARY_DATASET_NAMES if (PROCESSED_DATA_DIR / f"{dataset_name}_clean.parquet").exists()]


def build_experiment_frame(cleaned_df: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach metrics features to a cleaned dataset and retain key identifiers."""
    feature_df, metadata = build_metrics_features(cleaned_df, metrics, return_metadata=True)
    experiment_df = cleaned_df[[col for col in ["module_id", "project_name", "label", "commit_text"] if col in cleaned_df.columns]].copy()
    experiment_df = pd.concat([experiment_df, feature_df], axis=1)
    return experiment_df, metadata


def _safe_split(df: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame with stratification when possible, otherwise use a plain random split."""
    if df.empty:
        return df.copy(), df.copy()

    stratify_labels = df["label"] if "label" in df.columns and df["label"].value_counts(dropna=False).min() >= 2 else None
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify_labels)
    return train_df.copy(), test_df.copy()


def _build_train_val_test_splits(df: pd.DataFrame, test_size: float, val_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build reproducible train/val/test splits from one experiment frame."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1")

    holdout_size = test_size + val_size
    train_df, holdout_df = _safe_split(df, test_size=holdout_size, random_state=random_state)
    if holdout_df.empty:
        return train_df, holdout_df.copy(), holdout_df.copy()

    relative_val_size = val_size / holdout_size
    val_df, test_df = _safe_split(holdout_df, test_size=1 - relative_val_size, random_state=random_state)
    return train_df, val_df, test_df


def save_split_artifacts(dataset_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, manifest: dict[str, Any]) -> None:
    """Persist split identifiers and split metadata for one dataset."""
    dataset_split_dir = SPLITS_DIR / dataset_name
    save_split_ids(train_df, dataset_split_dir / "train_ids.csv")
    save_split_ids(val_df, dataset_split_dir / "val_ids.csv")
    save_split_ids(test_df, dataset_split_dir / "test_ids.csv")
    save_split_manifest(manifest, dataset_split_dir / "manifest.json")


def main() -> None:
    ensure_project_dirs()
    config = load_experiment_config()
    random_seed = int(config["random_seed"])
    test_size = float(config["test_size"])
    val_size = float(config["val_size"])
    metrics = list(config["metrics"])
    set_global_seed(random_seed)

    logger.info("Building experiment datasets with seed=%s, test_size=%s, val_size=%s", random_seed, test_size, val_size)

    experiment_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []

    for dataset_name in PRIMARY_DATASET_NAMES:
        cleaned_path = PROCESSED_DATA_DIR / f"{dataset_name}_clean.parquet"
        if not cleaned_path.exists():
            logger.warning("Skipping %s because %s does not exist.", dataset_name, cleaned_path)
            continue

        cleaned_df = read_parquet(cleaned_path)
        if cleaned_df.empty:
            logger.warning("Skipping %s because the cleaned dataset is empty.", dataset_name)
            continue

        experiment_df, feature_metadata = build_experiment_frame(cleaned_df, metrics)
        coverage = summarize_metric_coverage(cleaned_df, metrics)

        if "label" not in experiment_df.columns:
            raise ValueError(f"Cleaned dataset {dataset_name} is missing the label column")
        if "module_id" not in experiment_df.columns:
            raise ValueError(f"Cleaned dataset {dataset_name} is missing the module_id column")

        train_df, val_df, test_df = _build_train_val_test_splits(
            experiment_df,
            test_size=test_size,
            val_size=val_size,
            random_state=random_seed,
        )

        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError(f"Unable to construct non-empty splits for {dataset_name}")

        split_manifest = build_split_manifest(
            dataset_name=dataset_name,
            source_file=str(cleaned_path),
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            random_seed=random_seed,
            test_size=test_size,
            val_size=val_size,
            stratify_enabled=True,
        )
        save_split_artifacts(dataset_name, train_df, val_df, test_df, split_manifest)

        train_path = EXPERIMENT_DIR / f"{dataset_name}_train.parquet"
        val_path = EXPERIMENT_DIR / f"{dataset_name}_val.parquet"
        test_path = EXPERIMENT_DIR / f"{dataset_name}_test.parquet"
        write_parquet(train_df, train_path)
        write_parquet(val_df, val_path)
        write_parquet(test_df, test_path)

        experiment_path = EXPERIMENT_DIR / f"{dataset_name}_experiment.parquet"
        write_parquet(experiment_df, experiment_path)

        manifest_row = {
            "dataset_name": dataset_name,
            "source_file": str(cleaned_path),
            "experiment_file": str(experiment_path),
            "manifest_path": str(SPLITS_DIR / dataset_name / "manifest.json"),
            "num_rows": int(len(experiment_df)),
            "num_train_rows": int(len(train_df)),
            "num_val_rows": int(len(val_df)),
            "num_test_rows": int(len(test_df)),
            "feature_set": "metrics",
            "selected_metrics": ",".join(feature_metadata.get("selected_metrics", [])),
            "missing_metrics": ",".join(feature_metadata.get("missing_metrics", [])),
            "coverage_ratio": coverage["coverage_ratio"],
            "random_seed": random_seed,
            "test_size": test_size,
            "val_size": val_size,
        }
        experiment_records.append(manifest_row)
        summary_records.append({
            "dataset_name": dataset_name,
            "num_rows": int(len(experiment_df)),
            "num_defective": int(experiment_df["label"].value_counts().to_dict().get(1, 0)),
            "num_clean": int(experiment_df["label"].value_counts().to_dict().get(0, 0)),
            "num_features": int(feature_metadata.get("num_features", 0)),
            "coverage_ratio": coverage["coverage_ratio"],
        })

        logger.info("Built experiment dataset for %s at %s", dataset_name, experiment_path)

    write_csv(pd.DataFrame(experiment_records), MANIFEST_PATH)
    write_csv(pd.DataFrame(summary_records), SUMMARY_PATH)
    logger.info("Saved experiment manifest to %s", MANIFEST_PATH)
    logger.info("Saved experiment summary to %s", SUMMARY_PATH)


if __name__ == "__main__":
    main()
