"""Run the initial feature engineering scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import load_project_config
from src.data.clean import clean_dataset
from src.data.ingest import discover_raw_dataset_files, load_dataset
from src.data.unify_schema import unify_schema
from src.features.metrics_features import build_metrics_features, summarize_metric_coverage
from src.utils.io import write_csv, write_parquet
from src.utils.logging import get_logger
from src.utils.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_project_dirs

logger = get_logger(__name__)

PRIMARY_DATASET_FILES = {
    (PROJECT_ROOT / "data/raw/Promise + BPD/jm1.csv").resolve(),
    (PROJECT_ROOT / "data/raw/Promise + BPD/kc1.csv").resolve(),
    (PROJECT_ROOT / "data/raw/Promise + BPD/cm1.csv").resolve(),
    (PROJECT_ROOT / "data/raw/Promise + BPD/pc1.csv").resolve(),
}

LEGACY_PRIMARY_DATASET_FILES = {
    (PROJECT_ROOT / "data/raw/Promise/jm1.arff").resolve(),
    (PROJECT_ROOT / "data/raw/Promise/kc1.arff").resolve(),
    (PROJECT_ROOT / "data/raw/Promise/cm1.csv").resolve(),
    (PROJECT_ROOT / "data/raw/Promise/pc1.csv").resolve(),
}

ALL_PRIMARY_DATASET_FILES = PRIMARY_DATASET_FILES | LEGACY_PRIMARY_DATASET_FILES


def is_primary_dataset(file_path: Path) -> bool:
    """Return True when a raw file is one of the baseline datasets."""
    return file_path.resolve() in ALL_PRIMARY_DATASET_FILES


def discover_primary_dataset_files(raw_files: list[Path]) -> list[Path]:
    """Filter discovered files to the final baseline set."""
    return [file_path for file_path in raw_files if is_primary_dataset(file_path)]


def load_metrics_from_config() -> list[str]:
    """Load the list of core metric columns from the project config."""
    config = load_project_config()
    return config.get("features", {}).get("metrics", [])


def main() -> None:
    ensure_project_dirs()
    metrics_columns = load_metrics_from_config()
    raw_files = discover_raw_dataset_files(RAW_DATA_DIR)
    primary_raw_files = discover_primary_dataset_files(raw_files)
    logger.info("Feature pipeline discovered %s raw dataset file(s).", len(raw_files))
    logger.info("Feature pipeline selected %s primary dataset file(s).", len(primary_raw_files))

    coverage_records: list[dict] = []
    feature_artifacts: list[Path] = []

    for file_path in primary_raw_files:
        try:
            raw_df = load_dataset(file_path)
            unified_df = unify_schema(raw_df, dataset_name=file_path.name)
            cleaned_df, _ = clean_dataset(unified_df, deduplicate_by_module_id=True, return_summary=True)

            feature_df, metadata = build_metrics_features(cleaned_df, metrics_columns, return_metadata=True)
            coverage = summarize_metric_coverage(cleaned_df, metrics_columns)
            coverage_records.append(
                {
                    "dataset_name": file_path.stem,
                    "source_file": str(file_path),
                    "num_rows": len(cleaned_df),
                    "num_features": metadata["num_features"],
                    "coverage_ratio": coverage["coverage_ratio"],
                    "available_metrics": ",".join(coverage["available_metrics"]),
                    "missing_metrics": ",".join(coverage["missing_metrics"]),
                    "error": "",
                }
            )

            output_path = PROCESSED_DATA_DIR / f"{file_path.stem}_metrics.parquet"
            write_parquet(feature_df, output_path)
            feature_artifacts.append(output_path)
            logger.info("Saved metrics features to %s", output_path)
        except Exception as exc:
            logger.exception("Failed to build metrics features for %s: %s", file_path, exc)
            coverage_records.append(
                {
                    "dataset_name": file_path.stem,
                    "source_file": str(file_path),
                    "num_rows": None,
                    "num_features": None,
                    "coverage_ratio": None,
                    "available_metrics": "",
                    "missing_metrics": ",".join(metrics_columns),
                    "error": str(exc),
                }
            )

    coverage_df = pd.DataFrame(coverage_records)
    coverage_path = INTERIM_DATA_DIR / "metrics_feature_coverage.csv"
    write_csv(coverage_df, coverage_path)
    logger.info("Saved metrics feature coverage to %s", coverage_path)
    logger.info("Feature pipeline completed with %s artifact(s).", len(feature_artifacts))


if __name__ == "__main__":
    main()
