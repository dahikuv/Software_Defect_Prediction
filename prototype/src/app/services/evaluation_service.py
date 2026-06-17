"""Evaluation artifact helpers for the MVC backend."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.app.services.dataset_service import BASELINE_DATASETS
from src.app.state import StatusMessage
from src.utils.coercion import coerce_bool
from src.utils.io import read_csv
from src.utils.paths import RESULTS_TABLES_DIR
from src.utils.provenance import artifact_uses_commit_text

FINAL_MODELS_PATH = RESULTS_TABLES_DIR / "final_models_by_dataset.csv"
BEST_MODELS_PATH = RESULTS_TABLES_DIR / "best_models_by_dataset.csv"
EVALUATION_SUMMARY_PATH = RESULTS_TABLES_DIR / "evaluation_summary.csv"
MODEL_RANKING_PATH = RESULTS_TABLES_DIR / "model_ranking.csv"


def read_optional_csv(path) -> pd.DataFrame:
    """Read a CSV artifact when present, else return an empty frame."""
    if not path.exists():
        return pd.DataFrame()
    return read_csv(path)


def load_best_models_table() -> pd.DataFrame:
    """Load the best-model summary table."""
    final_df = read_optional_csv(FINAL_MODELS_PATH)
    if not final_df.empty:
        return final_df
    return read_optional_csv(BEST_MODELS_PATH)


def load_final_models_table() -> pd.DataFrame:
    """Load the final-model selection table."""
    return read_optional_csv(FINAL_MODELS_PATH)


def load_evaluation_summary() -> pd.DataFrame:
    """Load the evaluation summary table."""
    return read_optional_csv(EVALUATION_SUMMARY_PATH)


def load_model_ranking() -> pd.DataFrame:
    """Load the full model ranking table."""
    return read_optional_csv(MODEL_RANKING_PATH)


def list_available_datasets() -> list[str]:
    """Return baseline datasets that are available in the best-model table."""
    best_df = load_best_models_table()
    if best_df.empty or "dataset_name" not in best_df.columns:
        return BASELINE_DATASETS

    seen = set(best_df["dataset_name"].astype(str))
    ordered = [dataset for dataset in BASELINE_DATASETS if dataset in seen]
    extras = [dataset for dataset in seen if dataset not in BASELINE_DATASETS]
    return ordered + sorted(extras) or BASELINE_DATASETS


def list_available_models_for_dataset(dataset_name: str) -> list[str]:
    """Return ranked model options for one dataset."""
    ranking_df = load_model_ranking()
    if ranking_df.empty:
        best_df = load_best_models_table()
        if best_df.empty:
            return []
        filtered = best_df[best_df["dataset_name"] == dataset_name]
        return list(dict.fromkeys(filtered["model"].astype(str).tolist()))

    filtered = ranking_df[ranking_df["dataset_name"] == dataset_name].copy()
    if "rank_within_dataset" in filtered.columns:
        filtered = filtered.sort_values("rank_within_dataset")
    return list(dict.fromkeys(filtered["model"].astype(str).tolist()))


def select_model_row(frame: pd.DataFrame, dataset_name: str, selected_model: str) -> pd.Series | None:
    """Return the best matching row for a dataset/model pair."""
    if frame.empty or not {"dataset_name", "model"}.issubset(frame.columns):
        return None

    filtered = frame[
        (frame["dataset_name"].astype(str) == str(dataset_name))
        & (frame["model"].astype(str) == str(selected_model))
    ].copy()
    if filtered.empty:
        return None

    filtered["_final_selected_sort"] = (
        filtered["is_final_selected"].map(coerce_bool).map(lambda value: 0 if value else 1)
        if "is_final_selected" in filtered.columns
        else 1
    )
    filtered["_selection_rank_sort"] = (
        pd.to_numeric(filtered["selection_rank"], errors="coerce").fillna(float("inf"))
        if "selection_rank" in filtered.columns
        else float("inf")
    )
    filtered["_rank_sort"] = (
        pd.to_numeric(filtered["rank_within_dataset"], errors="coerce").fillna(float("inf"))
        if "rank_within_dataset" in filtered.columns
        else float("inf")
    )
    filtered["_uses_commit_text_sort"] = filtered.apply(
        lambda row: 0 if artifact_uses_commit_text(row.to_dict()) else 1,
        axis=1,
    )
    filtered["_model_path_sort"] = (
        filtered["model_path"].fillna("").astype(str).str.strip().map(lambda value: 0 if value else 1)
        if "model_path" in filtered.columns
        else 1
    )
    filtered = filtered.sort_values(
        [
            "_final_selected_sort",
            "_selection_rank_sort",
            "_rank_sort",
            "_uses_commit_text_sort",
            "_model_path_sort",
        ],
        kind="mergesort",
    )
    return filtered.drop(
        columns=[
            "_final_selected_sort",
            "_selection_rank_sort",
            "_rank_sort",
            "_uses_commit_text_sort",
            "_model_path_sort",
        ],
        errors="ignore",
    ).iloc[0]


def extract_metric_summary(dataset_name: str, selected_model: str) -> dict[str, Any]:
    """Return a compact metric summary for the selected row."""
    final_df = load_final_models_table()
    row = select_model_row(final_df, dataset_name, selected_model)

    if row is None:
        ranking_df = load_model_ranking()
        row = select_model_row(ranking_df, dataset_name, selected_model)

    if row is None:
        eval_df = load_evaluation_summary()
        row = select_model_row(eval_df, dataset_name, selected_model)

    if row is None:
        return {}

    keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "num_features",
        "num_train_rows",
        "num_val_rows",
        "num_test_rows",
        "decision_threshold",
        "threshold_constraint_met",
        "threshold_precision_floor",
        "threshold_selection_strategy",
        "selection_metric_primary",
        "selection_metric_secondary",
        "selection_policy",
        "selected_reason",
        "selection_schema_version",
        "selection_data_source",
        "test_metrics_report_only",
        "training_mode",
        "gap_train_val_f1",
        "gap_train_test_f1",
        "feature_family",
        "feature_set",
        "text_feature_column",
        "commit_feature_columns",
        "uses_commit_text",
        "artifact_schema_version",
        "artifact_stage",
        "artifact_group_key",
        "artifact_id",
        "source_results_table",
    ]
    summary = {key: row[key] for key in keys if key in row.index and pd.notna(row[key])}
    return _normalize_provenance_fields(summary)


def get_model_artifact_metadata(dataset_name: str, selected_model: str) -> dict[str, Any]:
    """Return the artifact metadata for the selected model row."""
    final_df = load_final_models_table()
    row = select_model_row(final_df, dataset_name, selected_model)

    if row is None:
        ranking_df = load_model_ranking()
        row = select_model_row(ranking_df, dataset_name, selected_model)

    if row is None:
        best_df = read_optional_csv(BEST_MODELS_PATH)
        row = select_model_row(best_df, dataset_name, selected_model)

    if row is None:
        return {}

    keys = [
        "model_path",
        "dataset_name",
        "model",
        "feature_family",
        "feature_set",
        "text_feature_column",
        "commit_feature_columns",
        "uses_commit_text",
        "decision_threshold",
        "threshold_constraint_met",
        "threshold_precision_floor",
        "threshold_selection_strategy",
        "selection_metric_primary",
        "selection_metric_secondary",
        "selection_policy",
        "selected_reason",
        "selection_schema_version",
        "selection_data_source",
        "test_metrics_report_only",
        "training_mode",
        "gap_train_val_f1",
        "gap_train_test_f1",
        "artifact_schema_version",
        "artifact_stage",
        "artifact_created_at",
        "artifact_group_key",
        "artifact_id",
        "source_results_table",
    ]
    metadata = {key: row[key] for key in keys if key in row.index and pd.notna(row[key])}
    return _normalize_provenance_fields(metadata)


def is_model_artifact_verifiable(selected_model_row: dict[str, Any]) -> tuple[bool, str]:
    """Return whether the selected artifact has the minimum metadata to verify provenance."""
    if not selected_model_row:
        return False, "No model row is available for verification."
    if not selected_model_row.get("model_path"):
        return False, "model_path is missing."
    if not selected_model_row.get("artifact_schema_version"):
        return False, "artifact_schema_version is missing."
    if not selected_model_row.get("artifact_id"):
        return False, "artifact_id is missing."
    return True, "Artifact metadata is sufficient for dashboard verification."


def resolve_model_artifact_row(dataset_name: str, selected_model: str) -> dict[str, Any]:
    """Return a merged model row with summary metadata for one dataset/model pair."""
    artifact_row = get_model_artifact_metadata(dataset_name, selected_model)
    summary = extract_metric_summary(dataset_name, selected_model)
    merged = {**summary, **artifact_row}
    if "feature_family" not in merged and "feature_set" in merged:
        merged["feature_family"] = merged["feature_set"]
    return merged


def describe_model_selection(dataset_name: str, selected_model: str, selected_model_row: dict[str, Any]) -> StatusMessage:
    """Return availability metadata for the current model selection."""
    if not selected_model:
        return StatusMessage(
            available=False,
            message="No model was resolved for the selected dataset.",
            details={"dataset_name": dataset_name},
        )

    if not selected_model_row:
        return StatusMessage(
            available=False,
            message="Model selection is not backed by a ranking or summary row.",
            details={"dataset_name": dataset_name, "selected_model": selected_model},
        )

    model_path = selected_model_row.get("model_path")
    verifiable, verification_message = is_model_artifact_verifiable(selected_model_row)
    uses_commit_text = artifact_uses_commit_text(selected_model_row)
    commit_text_available = coerce_bool(selected_model_row.get("commit_text_available", False)) or uses_commit_text
    details = {
        "dataset_name": dataset_name,
        "selected_model": selected_model,
        "model_path": str(model_path) if model_path and not pd.isna(model_path) else "",
        "feature_family": selected_model_row.get("feature_family") or selected_model_row.get("feature_set"),
        "feature_set": selected_model_row.get("feature_set") or selected_model_row.get("feature_family"),
        "text_feature_column": selected_model_row.get("text_feature_column") or selected_model_row.get("commit_text_column"),
        "commit_text_available": commit_text_available,
        "uses_commit_text": uses_commit_text,
        "artifact_schema_version": selected_model_row.get("artifact_schema_version"),
        "artifact_id": selected_model_row.get("artifact_id"),
        "artifact_group_key": selected_model_row.get("artifact_group_key"),
        "source_results_table": selected_model_row.get("source_results_table"),
        "artifact_verifiable": verifiable,
        "selection_policy": selected_model_row.get("selection_policy"),
        "selected_reason": selected_model_row.get("selected_reason"),
    }

    if model_path and not pd.isna(model_path):
        return StatusMessage(
            available=verifiable,
            message="Model selection is available and verifiable." if verifiable else f"Model selection exists, but verification is incomplete: {verification_message}",
            details=details,
        )

    return StatusMessage(
        available=False,
        message="Model selection is not verifiable because model_path is missing.",
        details=details,
    )


def _normalize_feature_family(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _normalize_provenance_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with canonical metadata/provenance aliases filled in."""
    normalized = dict(payload)
    feature_family = _normalize_feature_family(normalized.get("feature_family") or normalized.get("feature_set"))
    if feature_family:
        normalized["feature_family"] = feature_family
        normalized["feature_set"] = normalized.get("feature_set") or feature_family

    text_feature_column = normalized.get("text_feature_column") or normalized.get("commit_text_column")
    if text_feature_column:
        normalized["text_feature_column"] = text_feature_column
        normalized.setdefault("commit_text_column", text_feature_column)

    commit_feature_columns = normalized.get("commit_feature_columns")
    if commit_feature_columns is not None and not isinstance(commit_feature_columns, list):
        normalized["commit_feature_columns"] = [commit_feature_columns] if commit_feature_columns else []

    normalized["uses_commit_text"] = artifact_uses_commit_text(normalized)
    normalized["commit_text_available"] = coerce_bool(normalized.get("commit_text_available")) or normalized["uses_commit_text"]

    for key in ("artifact_schema_version", "artifact_stage", "artifact_group_key", "artifact_id", "source_results_table"):
        value = normalized.get(key)
        if value is not None and not pd.isna(value):
            normalized[key] = value
    return normalized


def row_to_dict(row: pd.Series | None) -> dict[str, Any]:
    """Convert a pandas row to a plain dict."""
    if row is None:
        return {}
    return _normalize_provenance_fields({key: value for key, value in row.to_dict().items() if pd.notna(value)})


__all__ = [
    "describe_model_selection",
    "extract_metric_summary",
    "get_model_artifact_metadata",
    "is_model_artifact_verifiable",
    "list_available_datasets",
    "list_available_models_for_dataset",
    "load_best_models_table",
    "load_evaluation_summary",
    "load_final_models_table",
    "load_model_ranking",
    "resolve_model_artifact_row",
    "row_to_dict",
    "select_model_row",
]
