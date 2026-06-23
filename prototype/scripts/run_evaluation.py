"""Run the evaluation scaffold."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_project_config
from src.evaluation.compare import (
    SUPPORTED_SELECTION_POLICIES,
    build_comparison_table,
    build_results_table,
    rank_models_by_dataset,
    select_final_models,
    summarize_results_table,
)
from src.evaluation.evaluate_significance import write_significance_table
from src.utils.io import read_csv, write_csv, write_json
from src.utils.logging import get_logger
from src.utils.paths import RESULTS_TABLES_DIR, ensure_project_dirs
from src.utils.provenance import artifact_uses_commit_text

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
HYBRID_RESULTS_PATH = RESULTS_TABLES_DIR / "hybrid_tfidf_results.csv"
FINAL_SELECTION_POLICY = "hybrid_validation_then_tuned"
DEFAULT_SIGNIFICANCE_BOOTSTRAP_ITERS = 300
DEFAULT_SIGNIFICANCE_PERMUTATION_ITERS = 300


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


def _coerce_bool_series(series: Any) -> Any:
    if getattr(series, "dtype", None) == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def _enrich_artifact_metadata(df: Any, stage_name: str, source_results_table: Path) -> Any:
    """Attach artifact-centric metadata columns when possible."""
    if df is None or getattr(df, "empty", True):
        return df

    enriched = df.copy()
    if "feature_family" not in enriched.columns and "feature_set" in enriched.columns:
        enriched["feature_family"] = enriched["feature_set"]
    if "feature_family" not in enriched.columns:
        enriched["feature_family"] = "metrics_only"
    else:
        enriched["feature_family"] = enriched["feature_family"].fillna("metrics_only")
    if "feature_set" not in enriched.columns and "feature_family" in enriched.columns:
        enriched["feature_set"] = enriched["feature_family"]
    elif "feature_set" in enriched.columns:
        enriched["feature_set"] = enriched["feature_set"].fillna(enriched["feature_family"])

    if "text_feature_column" not in enriched.columns:
        enriched["text_feature_column"] = ""
    if "uses_commit_text" not in enriched.columns:
        enriched["uses_commit_text"] = False
    else:
        enriched["uses_commit_text"] = _coerce_bool_series(enriched["uses_commit_text"])
    enriched["uses_commit_text"] = enriched.apply(lambda row: artifact_uses_commit_text(row.to_dict()), axis=1)

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


def _resolve_final_selection_policy(config: dict[str, Any]) -> str:
    evaluation_cfg = config.get("evaluation", {}) or {}
    return str(evaluation_cfg.get("selection_policy", FINAL_SELECTION_POLICY))


def _validate_evaluation_config(config: dict[str, Any]) -> None:
    selection_policy = _resolve_final_selection_policy(config)
    if selection_policy not in SUPPORTED_SELECTION_POLICIES:
        raise ValueError(
            f"Unsupported evaluation.selection_policy={selection_policy!r}; "
            f"expected one of {SUPPORTED_SELECTION_POLICIES}."
        )

    hybrid_cfg = config.get("features", {}).get("hybrid", {}) or {}
    for key in ("datasets", "final_selection_excluded"):
        value = hybrid_cfg.get(key, [])
        if value is not None and not isinstance(value, list):
            raise ValueError(f"features.hybrid.{key} must be a list when provided.")


def _resolve_final_hybrid_datasets(config: dict[str, Any]) -> set[str]:
    hybrid_cfg = config.get("features", {}).get("hybrid", {}) or {}
    explicit = hybrid_cfg.get("final_selection_datasets")
    if explicit:
        return {str(dataset).strip().lower() for dataset in explicit if str(dataset).strip()}

    configured = {
        str(dataset).strip().lower()
        for dataset in hybrid_cfg.get("datasets", [])
        if str(dataset).strip()
    }
    excluded = {
        str(dataset).strip().lower()
        for dataset in hybrid_cfg.get("final_selection_excluded", [])
        if str(dataset).strip()
    }
    return configured - excluded


def _filter_final_eligible_hybrid_candidates(df: Any, final_hybrid_datasets: set[str]) -> Any:
    """Keep hybrid rows that can participate in final model selection."""
    if df is None or getattr(df, "empty", True):
        return df
    if "dataset_name" not in df.columns:
        return df.iloc[0:0].copy()
    eligible = df[df["dataset_name"].astype(str).str.lower().isin(final_hybrid_datasets)].copy()
    if "model_path" in eligible.columns:
        eligible = eligible[eligible["model_path"].fillna("").astype(str).map(lambda value: bool(value) and Path(value).exists())]
    return eligible


def _with_training_mode(df: Any, training_mode: str) -> Any:
    if df is None or getattr(df, "empty", True):
        return df
    frame = df.copy()
    if "training_mode" not in frame.columns:
        frame["training_mode"] = training_mode
    else:
        frame["training_mode"] = frame["training_mode"].fillna(training_mode)
        frame.loc[frame["training_mode"].astype(str).str.strip().eq(""), "training_mode"] = training_mode
    return frame


def _build_unified_model_ranking(*frames: Any) -> pd.DataFrame:
    """Rank baseline, tuned, and hybrid candidates in one candidate table."""
    non_empty = [
        frame.drop(columns=["rank_within_dataset", "selection_rank", "is_final_selected"], errors="ignore").copy()
        for frame in frames
        if frame is not None and not getattr(frame, "empty", True)
    ]
    if not non_empty:
        return pd.DataFrame()
    combined = pd.concat(non_empty, ignore_index=True, sort=False)
    return rank_models_by_dataset(combined)


def _load_significance_config(config: dict[str, Any] | None = None) -> dict[str, int]:
    config = config or load_project_config()
    significance_cfg = config.get("evaluation", {}).get("significance", {}) or {}
    return {
        "n_bootstrap_iters": int(significance_cfg.get("bootstrap_iters", DEFAULT_SIGNIFICANCE_BOOTSTRAP_ITERS)),
        "n_permutation_iters": int(significance_cfg.get("permutation_iters", DEFAULT_SIGNIFICANCE_PERMUTATION_ITERS)),
    }


def main() -> None:
    ensure_project_dirs()
    config = load_project_config()
    _validate_evaluation_config(config)
    final_selection_policy = _resolve_final_selection_policy(config)
    final_hybrid_datasets = _resolve_final_hybrid_datasets(config)
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

    baseline_results_records = _safe_read_table(load_optional_table(BASELINE_RESULTS_PATH) or RESULTS_TABLE_PATH)
    tuned_results_records = _safe_read_table(TUNED_RESULTS_PATH)
    tuned_best_records = _safe_read_table(TUNED_BEST_PATH)
    hybrid_results_records = _safe_read_table(HYBRID_RESULTS_PATH)

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
    hybrid_results_df = _enrich_artifact_metadata(
        build_results_table(hybrid_results_records),
        stage_name="hybrid_tfidf_results",
        source_results_table=HYBRID_RESULTS_PATH,
    )
    final_hybrid_df = _filter_final_eligible_hybrid_candidates(hybrid_results_df, final_hybrid_datasets)
    unified_ranking_df = _enrich_artifact_metadata(
        build_results_table(
            _build_unified_model_ranking(
                _with_training_mode(ranked_df, "baseline"),
                _with_training_mode(tuned_best_df, "tuned"),
                _with_training_mode(final_hybrid_df, "hybrid_tfidf"),
            )
        ),
        stage_name="model_ranking",
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
        select_final_models(
            baseline_best_df=best_df,
            tuned_best_df=tuned_best_df,
            hybrid_best_df=final_hybrid_df,
            selection_policy=final_selection_policy,
        ),
        stage_name="final_models_by_dataset",
        source_results_table=results_path,
    )

    write_csv(unified_ranking_df, MODEL_RANKING_PATH)
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
            "hybrid_results_table": str(HYBRID_RESULTS_PATH),
            "final_hybrid_datasets": sorted(final_hybrid_datasets),
            "selection_policy": final_selection_policy,
            "selection_data_source": "cross_validation_or_validation",
            "test_metrics_report_only": True,
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

    significance_kwargs = _load_significance_config(config)
    significance_path = write_significance_table(**significance_kwargs)
    if significance_path is not None:
        logger.info("Saved significance table to %s", significance_path)
    else:
        logger.warning("Significance table not produced; check final selection and bundle availability.")


if __name__ == "__main__":
    main()
