"""Run the initial data pipeline scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import load_project_config
from src.data.clean import clean_dataset
from src.data.ingest import classify_dataset_file, discover_raw_dataset_files, load_dataset, profile_dataset, selection_note
from src.data.unify_schema import unify_schema
from src.data.validate import validate_required_columns
from src.utils.io import write_csv
from src.utils.logging import get_logger
from src.utils.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_project_dirs

logger = get_logger(__name__)


def load_metrics_from_config() -> list[str]:
    """Load the list of core metric columns from the project config."""
    config = load_project_config()
    return config.get("features", {}).get("metrics", [])


def main() -> None:
    ensure_project_dirs()
    metrics_columns = load_metrics_from_config()
    raw_files = discover_raw_dataset_files(RAW_DATA_DIR)
    logger.info("Discovered %s raw dataset file(s) in %s", len(raw_files), RAW_DATA_DIR)

    if not raw_files:
        empty_inventory = pd.DataFrame(
            columns=[
                "dataset_name",
                "source_file",
                "format",
                "num_rows_raw",
                "num_columns_raw",
                "num_rows_clean",
                "num_columns_clean",
                "num_modules",
                "num_defective",
                "num_clean",
                "imbalance_ratio",
                "defect_rate",
                "has_metrics",
                "has_commit_text",
                "has_project_name",
                "status",
                "notes",
            ]
        )
        inventory_path = INTERIM_DATA_DIR / "dataset_inventory.csv"
        write_csv(empty_inventory, inventory_path)
        logger.info("No raw dataset files found. Add datasets under data/raw/ and rerun this script.")
        logger.info("Saved empty dataset inventory to %s", inventory_path)
        return

    inventory_records: list[dict] = []
    clean_summary_records: list[dict] = []

    for file_path in raw_files:
        logger.info("Processing %s", file_path)
        try:
            raw_df = load_dataset(file_path)
            unified_df = unify_schema(raw_df, dataset_name=file_path.name)
            cleaned_df, summary = clean_dataset(
                unified_df,
                deduplicate_by_module_id=True,
                return_summary=True,
            )
            validate_required_columns(cleaned_df)

            output_path = PROCESSED_DATA_DIR / f"{Path(file_path).stem}_clean.parquet"
            cleaned_df.to_parquet(output_path, index=False)
            logger.info("Saved cleaned dataset to %s", output_path)

            inventory_records.append(
                profile_dataset(
                    raw_df=raw_df,
                    cleaned_df=cleaned_df,
                    dataset_name=file_path.stem,
                    source_file=file_path,
                    metrics_columns=metrics_columns,
                    status="ok",
                    notes=selection_note(file_path),
                )
            )
            clean_summary_records.append({"dataset_name": file_path.stem, **summary})
            logger.info("Cleaning summary for %s: %s", file_path.name, summary)
        except Exception as exc:
            logger.exception("Failed to process %s: %s", file_path, exc)
            dataset_tier, is_primary, is_supplementary, _, _ = classify_dataset_file(file_path)
            inventory_records.append(
                {
                    "dataset_name": file_path.stem,
                    "dataset_tier": dataset_tier,
                    "is_primary": is_primary,
                    "is_supplementary": is_supplementary,
                    "source_file": str(file_path),
                    "format": file_path.suffix.lower(),
                    "num_rows_raw": None,
                    "num_columns_raw": None,
                    "num_rows_clean": None,
                    "num_columns_clean": None,
                    "num_modules": None,
                    "num_defective": None,
                    "num_clean": None,
                    "imbalance_ratio": None,
                    "defect_rate": None,
                    "has_metrics": None,
                    "has_commit_text": None,
                    "has_project_name": None,
                    "status": "error",
                    "notes": f"{selection_note(file_path)}; {str(exc)}",
                }
            )

    inventory_df = pd.DataFrame(inventory_records)
    inventory_path = INTERIM_DATA_DIR / "dataset_inventory.csv"
    write_csv(inventory_df, inventory_path)
    logger.info("Saved dataset inventory to %s", inventory_path)
    logger.info("Exported all discovered dataset tiers, including candidate_primary rows, for visibility.")

    clean_summary_df = pd.DataFrame(clean_summary_records)
    clean_summary_path = INTERIM_DATA_DIR / "clean_summary.csv"
    write_csv(clean_summary_df, clean_summary_path)
    logger.info("Saved clean summary to %s", clean_summary_path)

    logger.info("Data pipeline pass completed.")
    logger.info("TODO: add dataset-specific adapters for PROMISE/NASA/AEEEM/GitHub edge cases.")


if __name__ == "__main__":
    main()
