"""Run the evaluation scaffold."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.compare import build_comparison_table, build_results_table, rank_models_by_dataset, select_final_models, summarize_results_table
from src.utils.io import read_csv, write_csv, write_json
from src.utils.logging import get_logger
from src.utils.paths import RESULTS_TABLES_DIR, ensure_project_dirs

logger = get_logger(__name__)
RESULTS_TABLE_PATH = RESULTS_TABLES_DIR / "results_table.csv"
EVALUATION_SUMMARY_PATH = RESULTS_TABLES_DIR / "evaluation_summary.csv"
BEST_MODEL_PATH = RESULTS_TABLES_DIR / "best_models_by_dataset.csv"
MODEL_RANKING_PATH = RESULTS_TABLES_DIR / "model_ranking.csv"
FINAL_SELECTION_PATH = RESULTS_TABLES_DIR / "final_models_by_dataset.csv"
FINAL_SELECTION_REPORT_PATH = RESULTS_TABLES_DIR / "final_selection_report.csv"
FINAL_SELECTION_META_PATH = RESULTS_TABLES_DIR / "final_selection_meta.json"
BASELINE_TUNED_COMPARISON_PATH = RESULTS_TABLES_DIR / "baseline_vs_tuned_comparison.csv"
BASELINE_RESULTS_PATH = RESULTS_TABLES_DIR / "baseline_results_table.csv"
TUNED_RESULTS_PATH = RESULTS_TABLES_DIR / "metrics_tuned_results.csv"
TUNED_BEST_PATH = RESULTS_TABLES_DIR / "metrics_tuned_best.csv"


def load_results_table() -> Path:
    """Return the path to the metrics-only results table if it exists."""
    if not RESULTS_TABLE_PATH.exists():
        raise FileNotFoundError(f"Missing results table: {RESULTS_TABLE_PATH}")
    return RESULTS_TABLE_PATH


def load_optional_table(path: Path) -> Path | None:
    if path.exists():
        return path
    return None


def _safe_read_table(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = read_csv(path)
    if df.empty:
        return []
    return df.to_dict("records")


def _enrich_artifact_metadata(df: Any, stage_name: str, source_results_table: Path) -> Any:
    """Attach artifact-centric metadata columns when possible."""
    if df is None or getattr(df, "empty", True):
        return df

    enriched = df.copy()
    if "feature_family" not in enriched.columns and "feature_set" in enriched.columns:
        enriched["feature_family"] = enriched["feature_set"]
    if "feature_set" not in enriched.columns and "feature_family" in enriched.columns:
        enriched["feature_set"] = enriched["feature_family"]

    if "text_feature_column" not in enriched.columns:
        enriched["text_feature_column"] = ""
    if "uses_commit_text" not in enriched.columns:
        enriched["uses_commit_text"] = False
    if "feature_family" in enriched.columns:
        enriched["uses_commit_text"] = enriched["uses_commit_text"].astype(bool) | enriched["feature_family"].astype(str).str.contains("commit", case=False, na=False)
    elif "text_feature_column" in enriched.columns:
        enriched["uses_commit_text"] = enriched["text_feature_column"].astype(str).str.strip().ne("")

    if "artifact_schema_version" not in enriched.columns:
        enriched["artifact_schema_version"] = "paper-v1"
    if "artifact_stage" not in enriched.columns:
        enriched["artifact_stage"] = stage_name
    if "artifact_created_at" not in enriched.columns:
        enriched["artifact_created_at"] = datetime.now().isoformat(timespec="seconds")
    if "source_results_table" not in enriched.columns:
        enriched["source_results_table"] = str(source_results_table)

    if "artifact_group_key" not in enriched.columns and {"dataset_name", "model"}.issubset(enriched.columns):
        enriched["artifact_group_key"] = enriched["dataset_name"].astype(str) + "::" + enriched["model"].astype(str)
    if "artifact_id" not in enriched.columns:
        if {"dataset_name", "model", "artifact_stage"}.issubset(enriched.columns):
            enriched["artifact_id"] = (
                enriched["dataset_name"].astype(str)
                + "::"
                + enriched["model"].astype(str)
                + "::"
                + enriched["artifact_stage"].astype(str)
            )
        else:
            enriched["artifact_id"] = stage_name
    return enriched


def main() -> None:
    ensure_project_dirs()
    results_path = load_results_table()
    logger.info("Loading results table from %s", results_path)

    results_df = build_results_table(_safe_read_table(results_path))
    if results_df.empty:
        logger.info("Results table is empty; nothing to evaluate.")
        return

    results_df = _enrich_artifact_metadata(results_df, stage_name="results_table", source_results_table=results_path)
    summary_df = _enrich_artifact_metadata(
        summarize_results_table(results_df),
        stage_name="evaluation_summary",
        source_results_table=results_path,
    )
    ranked_df = _enrich_artifact_metadata(
        rank_models_by_dataset(results_df),
        stage_name="model_ranking",
        source_results_table=results_path,
    )
    best_df = _enrich_artifact_metadata(
        ranked_df[ranked_df["rank_within_dataset"] == 1].copy(),
        stage_name="best_models_by_dataset",
        source_results_table=results_path,
    )

    write_csv(summary_df, EVALUATION_SUMMARY_PATH)
    write_csv(best_df, BEST_MODEL_PATH)
    write_csv(ranked_df, MODEL_RANKING_PATH)

    baseline_results_records = _safe_read_table(load_optional_table(BASELINE_RESULTS_PATH) or RESULTS_TABLE_PATH)
    tuned_results_records = _safe_read_table(TUNED_RESULTS_PATH)
    tuned_best_records = _safe_read_table(TUNED_BEST_PATH)

    baseline_results_df = _enrich_artifact_metadata(
        build_results_table(baseline_results_records),
        stage_name="baseline_results_table",
        source_results_table=results_path,
    )
    tuned_results_df = _enrich_artifact_metadata(
        build_results_table(tuned_results_records),
        stage_name="tuned_results_table",
        source_results_table=results_path,
    )
    tuned_best_df = _enrich_artifact_metadata(
        build_results_table(tuned_best_records),
        stage_name="tuned_best_table",
        source_results_table=results_path,
    )

    comparison_df = _enrich_artifact_metadata(
        build_comparison_table(
            baseline_df=best_df,
            tuned_df=tuned_best_df if not tuned_best_df.empty else best_df,
        ),
        stage_name="baseline_vs_tuned_comparison",
        source_results_table=results_path,
    )
    final_df = _enrich_artifact_metadata(
        select_final_models(best_df, tuned_best_df if not tuned_best_df.empty else best_df),
        stage_name="final_models_by_dataset",
        source_results_table=results_path,
    )

    write_csv(comparison_df, BASELINE_TUNED_COMPARISON_PATH)
    write_csv(final_df, FINAL_SELECTION_PATH)
    write_csv(final_df, FINAL_SELECTION_REPORT_PATH)
    write_json(
        {
            "source_results_table": str(results_path),
            "evaluation_summary": str(EVALUATION_SUMMARY_PATH),
            "best_models_by_dataset": str(BEST_MODEL_PATH),
            "model_ranking": str(MODEL_RANKING_PATH),
            "final_models_by_dataset": str(FINAL_SELECTION_PATH),
            "baseline_vs_tuned_comparison": str(BASELINE_TUNED_COMPARISON_PATH),
            "baseline_results_table": str(BASELINE_RESULTS_PATH),
            "tuned_results_table": str(TUNED_RESULTS_PATH),
            "tuned_best_table": str(TUNED_BEST_PATH),
            "artifact_schema_version": "paper-v1",
            "artifact_created_at": datetime.now().isoformat(timespec="seconds"),
        },
        FINAL_SELECTION_META_PATH,
    )

    logger.info("Saved evaluation summary to %s", EVALUATION_SUMMARY_PATH)
    logger.info("Saved best-model summary to %s", BEST_MODEL_PATH)
    logger.info("Saved model ranking to %s", MODEL_RANKING_PATH)
    logger.info("Saved baseline-vs-tuned comparison to %s", BASELINE_TUNED_COMPARISON_PATH)
    logger.info("Saved final selection to %s", FINAL_SELECTION_PATH)


if __name__ == "__main__":
    main()
