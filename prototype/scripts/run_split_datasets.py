"""Create reproducible train/validation/test split artifacts for primary datasets."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from src.data.clean import clean_dataset
from src.data.ingest import discover_raw_dataset_files, load_dataset
from src.data.split import (
    build_split_manifest,
    build_split_record,
    reconstruct_split_frames,
    save_split_ids,
    save_split_manifest,
    save_split_summary,
    split_with_ids,
    validate_split_ids,
)
from src.data.unify_schema import unify_schema
from src.utils.io import write_csv
from src.utils.logging import get_logger
from src.utils.paths import CONFIG_PATH, RAW_DATA_DIR, SPLITS_DIR, ensure_project_dirs
from src.utils.seed import set_global_seed

logger = get_logger(__name__)
PRIMARY_DATASET_FILES = {
    (PROJECT_ROOT / "data/raw/Promise/jm1.arff").resolve(),
    (PROJECT_ROOT / "data/raw/Promise/kc1.arff").resolve(),
    (PROJECT_ROOT / "data/raw/Promise/cm1.csv").resolve(),
    (PROJECT_ROOT / "data/raw/Promise/pc1.csv").resolve(),
}
INDEX_ROWS: list[dict] = []


def load_project_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def is_primary_dataset(file_path: Path) -> bool:
    return file_path.resolve() in PRIMARY_DATASET_FILES


def discover_primary_dataset_files(raw_files: list[Path]) -> list[Path]:
    return [file_path for file_path in raw_files if is_primary_dataset(file_path)]


def write_reconstructed_splits(dataset_dir: Path, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    write_csv(train_df, dataset_dir / "train_rows.csv")
    write_csv(val_df, dataset_dir / "val_rows.csv")
    write_csv(test_df, dataset_dir / "test_rows.csv")


def main() -> None:
    ensure_project_dirs()
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_project_config()
    random_seed = int(config.get("project", {}).get("random_seed", 42))
    test_size = float(config.get("split", {}).get("test_size", 0.2))
    val_size = float(config.get("split", {}).get("val_size", 0.1))
    set_global_seed(random_seed)

    raw_files = discover_raw_dataset_files(RAW_DATA_DIR)
    primary_raw_files = discover_primary_dataset_files(raw_files)
    logger.info("Split pipeline discovered %s primary dataset file(s).", len(primary_raw_files))

    summary_records: list[dict] = []

    for file_path in primary_raw_files:
        dataset_name = file_path.stem
        try:
            raw_df = load_dataset(file_path)
            unified_df = unify_schema(raw_df, dataset_name=file_path.name)
            cleaned_df, _ = clean_dataset(unified_df, deduplicate_by_module_id=True, return_summary=True)

            train_df, val_df, test_df = split_with_ids(
                cleaned_df,
                label_col="label",
                test_size=test_size,
                val_size=val_size,
                random_state=random_seed,
            )
            validate_split_ids(train_df, val_df, test_df)

            dataset_dir = SPLITS_DIR / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            save_split_ids(train_df, dataset_dir / "train_ids.csv")
            save_split_ids(val_df, dataset_dir / "val_ids.csv")
            save_split_ids(test_df, dataset_dir / "test_ids.csv")
            write_reconstructed_splits(dataset_dir, train_df, val_df, test_df)

            manifest = build_split_manifest(
                dataset_name=dataset_name,
                source_file=str(file_path),
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                random_seed=random_seed,
                test_size=test_size,
                val_size=val_size,
                stratify_enabled=True,
            )
            save_split_manifest(manifest, dataset_dir / "split_manifest.json")
            INDEX_ROWS.append(
                {
                    "dataset_name": dataset_name,
                    "source_file": str(file_path),
                    "dataset_dir": str(dataset_dir),
                    "manifest_path": str(dataset_dir / "split_manifest.json"),
                    "train_ids_path": str(dataset_dir / "train_ids.csv"),
                    "val_ids_path": str(dataset_dir / "val_ids.csv"),
                    "test_ids_path": str(dataset_dir / "test_ids.csv"),
                    "manifest_hash": manifest["manifest_hash"],
                }
            )

            summary_records.extend(
                [
                    {**build_split_record(dataset_name, "train", train_df), "source_file": str(file_path)},
                    {**build_split_record(dataset_name, "val", val_df), "source_file": str(file_path)},
                    {**build_split_record(dataset_name, "test", test_df), "source_file": str(file_path)},
                ]
            )
            logger.info("Saved split artifacts for %s", dataset_name)
        except Exception as exc:
            logger.exception("Failed to create split artifacts for %s: %s", dataset_name, exc)
            summary_records.append({"dataset_name": dataset_name, "split_name": "error", "num_rows": 0, "num_clean": 0, "num_defective": 0, "source_file": str(file_path), "error": str(exc)})

    save_split_summary(summary_records, SPLITS_DIR / "split_summary.csv")
    if INDEX_ROWS:
        save_split_summary(INDEX_ROWS, SPLITS_DIR / "index.csv")
    logger.info("Saved split summary to %s", SPLITS_DIR / "split_summary.csv")
    logger.info("Saved split index to %s", SPLITS_DIR / "index.csv")


if __name__ == "__main__":
    main()


def reconstruct_for_dataset(dataset_name: str, cleaned_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reconstruct split frames from saved artifacts for one dataset."""
    dataset_dir = SPLITS_DIR / dataset_name
    return reconstruct_split_frames(
        cleaned_df,
        dataset_dir / "train_ids.csv",
        dataset_dir / "val_ids.csv",
        dataset_dir / "test_ids.csv",
    )
