"""Ingest the JITLine + DeepJIT replication datasets into the project schema.

This script joins the JITLine change-metric CSVs with the DeepJIT message
pickles to produce per-commit clean parquet artifacts that match the rest of
the prototype pipeline (`module_id`, `label`, `commit_text`, project name,
plus JITLine metric columns).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.clean import clean_dataset
from src.data.jitline_ingest import (
    JITLINE_METRIC_COLUMNS,
    JITLINE_PROJECTS,
    available_jitline_projects,
    build_jitline_dataset,
)
from src.data.unify_schema import unify_schema
from src.data.validate import validate_dataset_schema
from src.utils.io import write_csv, write_parquet
from src.utils.logging import get_logger
from src.utils.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, ensure_project_dirs

logger = get_logger(__name__)

JITLINE_INVENTORY_PATH = INTERIM_DATA_DIR / "jitline_ingest_summary.csv"

def ingest_one_project(project: str) -> dict:
    raw_df = build_jitline_dataset(project)
    unified_df = unify_schema(raw_df, dataset_name=project)
    validate_dataset_schema(unified_df)
    cleaned_df, summary = clean_dataset(unified_df, deduplicate_by_module_id=True, return_summary=True)
    if cleaned_df.empty:
        raise ValueError(f"JITLine ingest produced an empty cleaned frame for {project}")

    output_path = PROCESSED_DATA_DIR / f"{project}_clean.parquet"
    write_parquet(cleaned_df, output_path)
    label_counts = cleaned_df["label"].astype(int).value_counts().to_dict()
    metric_columns_present = [column for column in JITLINE_METRIC_COLUMNS if column in cleaned_df.columns]
    has_text = bool(cleaned_df["commit_text"].astype(str).str.strip().ne("").any())

    return {
        "dataset_name": project,
        "source_dir": "data/raw/JITLine",
        "output_file": str(output_path),
        "num_rows_raw": int(len(raw_df)),
        "num_rows_clean": int(len(cleaned_df)),
        "num_train_pkl_rows": int((raw_df["jitline_split"] == "train").sum()),
        "num_test_pkl_rows": int((raw_df["jitline_split"] == "test").sum()),
        "num_clean_label_0": int(label_counts.get(0, 0)),
        "num_clean_label_1": int(label_counts.get(1, 0)),
        "num_metric_columns": len(metric_columns_present),
        "metric_columns": ",".join(metric_columns_present),
        "has_commit_text": has_text,
        "module_duplicates_removed": int(summary.get("module_duplicates_removed", 0)),
        "rows_missing_label_removed": int(summary.get("rows_missing_label_removed", 0)),
    }

def main() -> None:
    ensure_project_dirs()
    available = available_jitline_projects()
    if not available:
        logger.warning(
            "JITLine raw files not found under data/raw/JITLine. Expected per-project files like %s.",
            ", ".join(f"{project}_metrics.csv" for project in JITLINE_PROJECTS),
        )
        write_csv(pd.DataFrame(columns=["dataset_name"]), JITLINE_INVENTORY_PATH)
        return

    records: list[dict] = []
    for project in available:
        logger.info("Ingesting JITLine project %s", project)
        record = ingest_one_project(project)
        records.append(record)
        logger.info(
            "Wrote %s (rows=%s, label_dist=%s/%s)",
            record["output_file"],
            record["num_rows_clean"],
            record["num_clean_label_0"],
            record["num_clean_label_1"],
        )

    write_csv(pd.DataFrame(records), JITLINE_INVENTORY_PATH)
    logger.info("Saved JITLine ingest summary to %s", JITLINE_INVENTORY_PATH)

if __name__ == "__main__":
    main()
