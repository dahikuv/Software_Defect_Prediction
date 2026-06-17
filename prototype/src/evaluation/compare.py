"""Comparison helpers for aggregating experiment results."""

from __future__ import annotations

import pandas as pd

from src.utils.coercion import coerce_bool
from src.utils.provenance import artifact_uses_commit_text

SUMMARY_METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "auc"]
SELECTION_METRIC_COLUMNS = ["recall", "f1", "auc", "precision", "accuracy"]
SELECTION_COLUMN_PRIORITY = {
    "recall": ["val_recall", "threshold_val_recall", "cv_mean_recall"],
    "f1": ["val_f1", "threshold_val_f1", "cv_mean_f1"],
    "auc": ["val_auc", "threshold_val_auc", "cv_mean_auc"],
    "precision": ["val_precision", "threshold_val_precision", "cv_mean_precision"],
    "accuracy": ["val_accuracy"],
}

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
    "num_val_rows",
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
    "decision_threshold",
    "threshold_precision_floor",
    "threshold_constraint_met",
    "threshold_selection_metric",
    "threshold_selection_strategy",
    "threshold_val_precision",
    "threshold_val_recall",
    "threshold_val_f1",
    "threshold_val_auc",
    "cv_folds",
    "cv_mean_precision",
    "cv_std_precision",
    "cv_mean_recall",
    "cv_std_recall",
    "cv_mean_f1",
    "cv_std_f1",
    "cv_mean_auc",
    "cv_std_auc",
    "selection_data_source",
    "test_metrics_report_only",
    "train_accuracy",
    "train_precision",
    "train_recall",
    "train_f1",
    "train_auc",
    "val_accuracy",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "gap_train_test_accuracy",
    "gap_train_test_precision",
    "gap_train_test_recall",
    "gap_train_test_f1",
    "gap_train_test_auc",
    "gap_train_val_accuracy",
    "gap_train_val_precision",
    "gap_train_val_recall",
    "gap_train_val_f1",
    "gap_train_val_auc",
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


def _safe_bool(value: object) -> bool:
    return coerce_bool(value)


def _add_threshold_sort_column(df: pd.DataFrame) -> pd.DataFrame:
    sorted_df = df.copy()
    if "threshold_constraint_met" in sorted_df.columns:
        sorted_df["_threshold_constraint_sort"] = sorted_df["threshold_constraint_met"].map(_safe_bool)
    else:
        sorted_df["_threshold_constraint_sort"] = True
    return sorted_df

def _selection_sort_series(
    df: pd.DataFrame,
    metric: str,
    allow_report_metric_fallback: bool = False,
) -> pd.Series:
    candidates = list(SELECTION_COLUMN_PRIORITY.get(metric, []))
    if allow_report_metric_fallback:
        candidates.append(metric)

    selected = pd.Series(float("-inf"), index=df.index, dtype=float)
    for column in candidates:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        selected = selected.mask((selected == float("-inf")) & values.notna(), values)
    return selected.fillna(float("-inf"))

def _selection_source_for_row(row: pd.Series) -> str:
    existing = row.get("selection_data_source")
    if existing is not None and not pd.isna(existing) and str(existing).strip():
        return str(existing)

    cv_columns = [column for columns in SELECTION_COLUMN_PRIORITY.values() for column in columns if column.startswith("cv_")]
    val_columns = [
        column
        for columns in SELECTION_COLUMN_PRIORITY.values()
        for column in columns
        if column.startswith("val_") or column.startswith("threshold_val_")
    ]
    if any(column in row.index and pd.notna(pd.to_numeric(row[column], errors="coerce")) for column in cv_columns):
        return "cross_validation"
    if any(column in row.index and pd.notna(pd.to_numeric(row[column], errors="coerce")) for column in val_columns):
        return "validation"
    return "unresolved"


def _sort_for_selection(df: pd.DataFrame, primary_metric: str = "recall", secondary_metric: str = "f1") -> pd.DataFrame:
    if df.empty or "dataset_name" not in df.columns:
        return df

    sorted_df = _add_threshold_sort_column(df)
    sort_columns = ["dataset_name", "_threshold_constraint_sort"]
    ascending = [True, False]

    for metric in [primary_metric, secondary_metric, "auc", "precision", "accuracy"]:
        sort_column = f"_selection_{metric}_sort"
        sorted_df[sort_column] = _selection_sort_series(sorted_df, metric, allow_report_metric_fallback=False)
        sort_columns.append(sort_column)
        ascending.append(False)

    for gap_metric in ["gap_train_val_f1", "gap_train_val_auc"]:
        if gap_metric in sorted_df.columns:
            sort_column = f"_{gap_metric}_abs_sort"
            sorted_df[sort_column] = pd.to_numeric(sorted_df[gap_metric], errors="coerce").abs().fillna(float("inf"))
            sort_columns.append(sort_column)
            ascending.append(True)

    sorted_df = sorted_df.sort_values(sort_columns, ascending=ascending, kind="mergesort").copy()
    helper_columns = [
        column
        for column in sorted_df.columns
        if column.startswith("_selection_") or column.endswith("_abs_sort")
    ]
    return sorted_df.drop(columns=["_threshold_constraint_sort", *helper_columns], errors="ignore")


def summarize_results_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize metrics by dataset and model for evaluation reporting."""
    if results_df.empty:
        return results_df
    if not {"dataset_name", "model"}.issubset(results_df.columns):
        return results_df

    metric_columns = [col for col in SUMMARY_METRIC_COLUMNS if col in results_df.columns]
    if not metric_columns:
        return results_df

    grouped = (
        results_df.groupby(["dataset_name", "model"], as_index=False)[metric_columns]
        .mean(numeric_only=True)
    )
    return _sort_for_selection(grouped)


def rank_models_by_dataset(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return models sorted within each dataset by recall-first selection criteria."""
    if results_df.empty:
        return results_df
    if "dataset_name" not in results_df.columns:
        return results_df

    ranked = _sort_for_selection(results_df)
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
    for column in key_columns + shared_columns:
        if column in baseline.columns:
            baseline[column] = baseline[column].fillna("").astype(str)
        if column in tuned.columns:
            tuned[column] = tuned[column].fillna("").astype(str)

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


SUPPORTED_SELECTION_POLICIES = ("tuned_first", "best_validation", "hybrid_validation_then_tuned")


def _filter_valid_hybrid_candidates(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Return hybrid rows that report fitted commit-text features."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    hybrid = _normalize_training_mode(frame, "hybrid_tfidf")
    if "uses_commit_text" not in hybrid.columns and not any(
        column in hybrid.columns for column in ["tfidf_num_features", "sbert_num_features", "embedding_dim", "num_commit_features"]
    ):
        return pd.DataFrame()
    mask = hybrid.apply(lambda row: artifact_uses_commit_text(row.to_dict()), axis=1)
    return hybrid.loc[mask].copy()


def select_final_models(
    baseline_best_df: pd.DataFrame,
    tuned_best_df: pd.DataFrame,
    hybrid_best_df: pd.DataFrame | None = None,
    selection_metric: str = "recall",
    secondary_metric: str = "f1",
    selection_policy: str = "tuned_first",
) -> pd.DataFrame:
    """Choose the final model per dataset from baseline, tuned, and hybrid candidates.

    selection_policy
    ----------------
    - ``"tuned_first"`` (default): always prefer the tuned best per dataset.
      Baselines fill in only when no tuned candidate exists for a dataset.
      This keeps the final selection comparable across datasets because every
      reported model went through the same CV-tuned grid.
    - ``"best_validation"``: original behaviour. Pool baseline + tuned, plus
      any valid hybrid rows passed by the caller, and pick the row with the
      strongest validation-driven sort key per dataset.
    - ``"hybrid_validation_then_tuned"``: pool tuned candidates, baseline
      fallback rows, and valid hybrid commit-text rows. Hybrid can win only
      when its validation-driven sort key beats the tuned/baseline candidate.

    The metric the sorter uses (recall then f1 by default) and its underlying
    column priority are unchanged across both policies.
    """
    if selection_policy not in SUPPORTED_SELECTION_POLICIES:
        raise ValueError(
            f"Unsupported selection_policy: {selection_policy!r}. "
            f"Expected one of {SUPPORTED_SELECTION_POLICIES}."
        )

    hybrid = _filter_valid_hybrid_candidates(hybrid_best_df)

    if baseline_best_df.empty and tuned_best_df.empty and hybrid.empty:
        return pd.DataFrame()

    baseline = _normalize_training_mode(baseline_best_df, "baseline")
    tuned = _normalize_training_mode(tuned_best_df, "tuned")

    if selection_policy == "tuned_first":
        # Prefer tuned per dataset; fall back to baseline only when tuned is missing.
        tuned_datasets = set(tuned["dataset_name"].astype(str)) if "dataset_name" in tuned.columns else set()
        baseline_only = (
            baseline[~baseline["dataset_name"].astype(str).isin(tuned_datasets)]
            if "dataset_name" in baseline.columns and tuned_datasets
            else baseline
        )
        candidates = pd.concat([tuned, baseline_only], ignore_index=True, sort=False)
    elif selection_policy == "best_validation":
        if not hybrid.empty:
            candidates = pd.concat([baseline, tuned, hybrid], ignore_index=True, sort=False)
        else:
            candidates = pd.concat([baseline, tuned], ignore_index=True, sort=False)
    else:
        # Keep tuned-first coverage, but let valid hybrid rows compete on the
        # same validation sort key. Baseline rows are only used when tuned is
        # absent for a dataset.
        tuned_datasets = set(tuned["dataset_name"].astype(str)) if "dataset_name" in tuned.columns else set()
        baseline_only = (
            baseline[~baseline["dataset_name"].astype(str).isin(tuned_datasets)]
            if "dataset_name" in baseline.columns and tuned_datasets
            else baseline
        )
        candidates = pd.concat([tuned, baseline_only, hybrid], ignore_index=True, sort=False)

    if candidates.empty:
        return candidates
    if "dataset_name" not in candidates.columns:
        return candidates

    candidates = _sort_for_selection(candidates, primary_metric=selection_metric, secondary_metric=secondary_metric).copy()
    candidates["selection_rank"] = candidates.groupby("dataset_name").cumcount() + 1
    candidates["is_final_selected"] = candidates["selection_rank"] == 1
    candidates["selection_metric_primary"] = selection_metric
    candidates["selection_metric_secondary"] = secondary_metric
    candidates["selection_policy"] = selection_policy
    candidates["selection_data_source"] = candidates.apply(_selection_source_for_row, axis=1)
    candidates["test_metrics_report_only"] = True
    candidates["selected_reason"] = candidates.apply(
        lambda row: (
            f"Selected by {selection_metric} then {secondary_metric} using "
            f"{row['selection_data_source']} metrics within dataset under policy "
            f"{selection_policy}"
        ),
        axis=1,
    )
    candidates["selection_schema_version"] = "paper-v1"
    final = candidates.loc[candidates["is_final_selected"]].copy()

    preferred_columns = [
        "dataset_name",
        "model",
        "training_mode",
        "selection_policy",
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
