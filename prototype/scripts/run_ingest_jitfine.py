"""Prepare the JIT-Fine cleaned dataset under prototype/data/processed."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.jitfine_ingest import (
    JITFINE_DATASET_NAME,
    build_jitfine_dataset,
    jitfine_artifacts_available,
    resolve_jitfine_feature_paths,
)
from src.utils.io import write_csv, write_parquet
from src.utils.logging import get_logger
from src.utils.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, ensure_project_dirs

logger = get_logger(__name__)


def main() -> None:
    ensure_project_dirs()
    if not jitfine_artifacts_available():
        paths = resolve_jitfine_feature_paths()
        missing = [str(path) for path in paths.values() if not path.exists()]
        logger.error("JIT-Fine feature pickles are missing: %s", missing)
        raise SystemExit(1)

    df = build_jitfine_dataset()
    output_path = PROCESSED_DATA_DIR / f"{JITFINE_DATASET_NAME}_clean.parquet"
    write_parquet(df, output_path)

    summary = (
        df.groupby(["jitline_split", "project_name"], dropna=False)
        .agg(num_rows=("module_id", "count"), num_defective=("label", "sum"))
        .reset_index()
    )
    summary["num_clean"] = summary["num_rows"] - summary["num_defective"]
    summary_path = INTERIM_DATA_DIR / "jitfine_ingest_summary.csv"
    write_csv(summary, summary_path)

    logger.info(
        "JIT-Fine ingest complete: rows=%s projects=%s split=%s",
        len(df),
        df["project_name"].nunique(),
        df["jitline_split"].value_counts().to_dict(),
    )
    logger.info("Saved JIT-Fine cleaned parquet to %s", output_path)
    logger.info("Saved JIT-Fine ingest summary to %s", summary_path)


if __name__ == "__main__":
    main()