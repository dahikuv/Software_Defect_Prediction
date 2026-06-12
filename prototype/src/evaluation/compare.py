"""Comparison helpers for aggregating experiment results."""

from __future__ import annotations

import pandas as pd

SUMMARY_METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "auc"]
SELECTION_METRIC_COLUMNS = ["auc", "f1", "recall", "precision", "accuracy"]

PREFERRED_RESULT_COLUMNS = [
    "dataset_name",
    "model",
    "feature_family",
    "feature_set",
    "text_feature_column",
    "commit_text_column",
    "commit_text_available",
    "uses_commit_text",
    "artifact_schema_version",
    "artifact_stage",
    "artifact_created_at",
    "artifact_group_key",
    "artifact_id",
    "source_results_table",
    "source_file",
    "random_seed",
    "test_size",
    "stratified_split",
    "stratify_enabled",
    "configured_models",
    "configured_metrics_count",
    "configured_metrics",
    "num_train_rows",
    "num_test_rows",
    "num_rows",
    "num_features",
    "num_clean",
    "num_defective",
    "train_num_clean",
    "train_num_defective",
    "test_num_clean",
    "test_num_defective",
    "train_label_distribution",
    "test_label_distribution",
    "selected_metrics",
    "missing_metrics",
    "dropped_all_nan_metrics",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "model_path",
    "stage",
    "error",
]

def _prefix_columns(frame: pd.DataFrame, prefix: str, keep: list[str] | None = None) -> pd.DataFrame:
    keep = keep or []
    renamed = frame.copy()
    rename_map = {col: f"{prefix}{col}" for col in renamed.columns if col not in keep}
    return renamed.rename(columns=rename_map)


def _as_dataframe(results: list[dict] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        return results.copy()
    return pd.DataFrame(results)


def build_results_table(results: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """Convert results to a stable, readable DataFrame."""
    df = _as_dataframe(results)
    if df.empty:
        return df

    preferred = [col for col in PREFERRED_RESULT_COLUMNS if col in df.columns]
    remaining = [col for col in df.columns if col not in preferred]
    return df[preferred + remaining]


def summarize_results_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize metrics by dataset and model for evaluation reporting."""
    if results_df.empty:
        return results_df

    grouped = (
        results_df.groupby(["dataset_name", "model"], as_index=False)[SUMMARY_METRIC_COLUMNS]
        .mean(numeric_only=True)
        .sort_values(["dataset_name", "auc", "f1"], ascending=[True, False, False])
    )
    return grouped


def rank_models_by_dataset(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return the best model per dataset sorted by AUC then F1."""
    if results_df.empty:
        return results_df

    ranked = results_df.sort_values(["dataset_name", "auc", "f1"], ascending=[True, False, False]).copy()
    ranked["rank_within_dataset"] = ranked.groupby("dataset_name").cumcount() + 1
    return ranked


def build_comparison_table(
    baseline_df: pd.DataFrame,
    tuned_df: pd.DataFrame,
    key_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build a side-by-side baseline vs tuned comparison table."""
    key_columns = key_columns or ["dataset_name", "model"]
    baseline = baseline_df.copy()
    tuned = tuned_df.copy()

    shared_metadata = [
        "feature_family",
        "feature_set",
        "text_feature_column",
        "commit_text_column",
        "commit_text_available",
        "uses_commit_text",
        "artifact_schema_version",
        "artifact_stage",
        "artifact_created_at",
        "artifact_group_key",
        "artifact_id",
        "source_results_table",
    ]
    shared_columns = [col for col in shared_metadata if col in baseline.columns and col in tuned.columns and col not in key_columns]

    baseline = _prefix_columns(baseline, "baseline_", keep=key_columns + shared_columns)
    tuned = _prefix_columns(tuned, "tuned_", keep=key_columns + shared_columns)

    comparison = baseline.merge(tuned, on=key_columns + shared_columns, how="outer")

    for metric in SELECTION_METRIC_COLUMNS:
        baseline_col = f"baseline_{metric}"
        tuned_col = f"tuned_{metric}"
        if baseline_col in comparison.columns and tuned_col in comparison.columns:
            comparison[f"delta_{metric}"] = comparison[tuned_col] - comparison[baseline_col]

    comparison["comparison_schema_version"] = "paper-v1"
    comparison["comparison_mode"] = "frozen_baseline_vs_tuned"
    return comparison


def _normalize_training_mode(frame: pd.DataFrame, default_mode: str) -> pd.DataFrame:
    normalized = frame.copy()
    if "training_mode" not in normalized.columns:
        normalized["training_mode"] = default_mode
    return normalized


def select_final_models(
    baseline_best_df: pd.DataFrame,
    tuned_best_df: pd.DataFrame,
    selection_metric: str = "auc",
    secondary_metric: str = "f1",
) -> pd.DataFrame:
    """Choose the final model per dataset from baseline and tuned candidates."""
    if baseline_best_df.empty and tuned_best_df.empty:
        return pd.DataFrame()

    baseline = _normalize_training_mode(baseline_best_df, "baseline")
    tuned = _normalize_training_mode(tuned_best_df, "tuned")
    baseline = _prefix_columns(baseline, "baseline_", keep=["dataset_name", "model", "training_mode"])
    tuned = _prefix_columns(tuned, "tuned_", keep=["dataset_name", "model", "training_mode"])

    candidates = pd.concat([baseline, tuned], ignore_index=True, sort=False)
    if candidates.empty:
        return candidates

    candidates = candidates.sort_values(
        ["dataset_name", selection_metric],
        ascending=[True, False],
        kind="mergesort",
    ).copy()
    candidates["selection_rank"] = candidates.groupby("dataset_name").cumcount() + 1
    candidates["is_final_selected"] = candidates["selection_rank"] == 1
    candidates["selection_metric_primary"] = selection_metric
    candidates["selection_metric_secondary"] = secondary_metric
    candidates["selected_reason"] = candidates.apply(
        lambda row: f"Selected by {selection_metric} then {secondary_metric} within dataset",
        axis=1,
    )
    candidates["selection_schema_version"] = "paper-v1"
    final = candidates.loc[candidates["is_final_selected"]].copy()

    preferred_columns = [
        "dataset_name",
        "model",
        "training_mode",
        "selection_rank",
        "is_final_selected",
        "selection_metric_primary",
        "selection_metric_secondary",
        "selected_reason",
        "selection_schema_version",
    ]
    remaining = [col for col in final.columns if col not in preferred_columns]
    return final[preferred_columns + remaining]


__all__ = [
    "build_comparison_table",
    "build_results_table",
    "rank_models_by_dataset",
    "select_final_models",
    "summarize_results_table",
    "PREFERRED_RESULT_COLUMNS",
]
