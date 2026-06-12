"""Model inference helpers for the MVC backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.app.state import StatusMessage
from src.features.metrics_features import build_commit_text_features
from src.models.predict import load_model, predict_with_model


def _build_prediction_input(sample_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Prepare the feature matrix used by the saved model."""
    prediction_input = sample_df.copy()
    for column in feature_columns:
        if column not in prediction_input.columns:
            prediction_input[column] = 0.0
    return prediction_input[feature_columns].copy()



def _resolve_feature_family(selected_model_row: dict[str, Any], default: str = "metrics_only") -> str:
    return str(selected_model_row.get("feature_family") or selected_model_row.get("feature_set") or default)


def _normalize_feature_family(selected_model_row: dict[str, Any]) -> str:
    return str(selected_model_row.get("feature_family") or selected_model_row.get("feature_set") or "metrics_only")



def _resolve_text_feature_column(selected_model_row: dict[str, Any], feature_details: dict[str, Any] | None = None) -> str | None:
    if feature_details and feature_details.get("text_feature_column"):
        return str(feature_details.get("text_feature_column"))
    text_feature_column = selected_model_row.get("text_feature_column")
    return str(text_feature_column) if text_feature_column else None


def _resolve_feature_columns(
    selected_model_row: dict[str, Any],
    sample_df: pd.DataFrame,
    metric_columns: list[str],
) -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    """Resolve feature columns for inference and synthesize commit-text features when needed."""
    feature_family = _normalize_feature_family(selected_model_row)
    resolved_df = sample_df.copy()
    resolved_metric_columns = [column for column in metric_columns if column in resolved_df.columns]
    feature_columns = list(resolved_metric_columns)
    details: dict[str, Any] = {
        "feature_family": feature_family,
        "metric_feature_columns": resolved_metric_columns,
        "commit_feature_columns": [],
        "generated_commit_features": False,
        "text_feature_column": selected_model_row.get("text_feature_column"),
    }

    needs_commit_features = feature_family in {"metrics_plus_commit_text", "metrics_plus_text", "hybrid", "commit_text_only"}
    if needs_commit_features:
        commit_df, commit_meta = build_commit_text_features(resolved_df, return_metadata=True)
        details["text_feature_column"] = commit_meta.get("text_column") or details["text_feature_column"]
        details["commit_feature_count"] = int(commit_meta.get("num_features", 0))
        if not commit_df.empty:
            commit_df = commit_df.add_prefix("commit_")
            commit_columns = list(commit_df.columns)
            resolved_df = pd.concat([resolved_df.reset_index(drop=True), commit_df.reset_index(drop=True)], axis=1)
            feature_columns.extend(commit_columns)
            details.update(
                {
                    "commit_feature_columns": commit_columns,
                    "generated_commit_features": True,
                }
            )
        elif feature_family == "commit_text_only":
            feature_columns = []

    if feature_family == "commit_text_only" and details["commit_feature_columns"]:
        feature_columns = list(details["commit_feature_columns"])

    feature_columns = list(dict.fromkeys(feature_columns))
    return feature_columns, resolved_df, details


def _extract_metadata(selected_model_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_name": selected_model_row.get("dataset_name"),
        "model": selected_model_row.get("model"),
        "feature_family": selected_model_row.get("feature_family") or selected_model_row.get("feature_set") or "metrics_only",
        "text_feature_column": selected_model_row.get("text_feature_column"),
        "model_path": selected_model_row.get("model_path"),
    }


def build_sample_predictions(
    selected_model_row: dict, sample_df: pd.DataFrame, metric_columns: list[str]
) -> tuple[list[dict], StatusMessage]:
    """Run predictions on sample rows using the saved model artifact."""
    model_path = selected_model_row.get("model_path")
    if not model_path:
        return [], StatusMessage(
            available=False,
            message="Prediction skipped because model_path is missing.",
            details=_extract_metadata(selected_model_row),
        )

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        return [], StatusMessage(
            available=False,
            message="Prediction skipped because the saved model artifact was not found.",
            details={**_extract_metadata(selected_model_row), "model_path": str(model_path_obj)},
        )

    if sample_df.empty:
        return [], StatusMessage(
            available=False,
            message="Prediction skipped because no sample rows are available.",
            details=_extract_metadata(selected_model_row),
        )

    if not metric_columns:
        return [], StatusMessage(
            available=False,
            message="Prediction skipped because no metric columns are available.",
            details=_extract_metadata(selected_model_row),
        )

    feature_columns, resolved_df, feature_details = _resolve_feature_columns(selected_model_row, sample_df, metric_columns)
    if not feature_columns:
        return [], StatusMessage(
            available=False,
            message="Prediction skipped because no usable feature columns were resolved.",
            details={**_extract_metadata(selected_model_row), **feature_details},
        )

    try:
        model = load_model(model_path_obj)
        prediction_input = _build_prediction_input(resolved_df, feature_columns)
        predictions_df = predict_with_model(model, prediction_input)
    except Exception as exc:
        return [], StatusMessage(
            available=False,
            message="Prediction failed while loading or running the saved model.",
            details={**_extract_metadata(selected_model_row), **feature_details, "error": str(exc)},
        )

    merged_df = pd.concat(
        [
            resolved_df[[column for column in ["module_id", "label"] if column in resolved_df.columns]].reset_index(drop=True),
            predictions_df.reset_index(drop=True),
        ],
        axis=1,
    )
    merged_df["feature_family"] = feature_details.get("feature_family", _normalize_feature_family(selected_model_row))
    merged_df["model_name"] = selected_model_row.get("model")
    merged_df["text_feature_column"] = feature_details.get("text_feature_column")
    merged_df["metric_feature_columns"] = ", ".join(feature_details.get("metric_feature_columns", []))
    merged_df["commit_feature_columns"] = ", ".join(feature_details.get("commit_feature_columns", []))
    return merged_df.to_dict("records"), StatusMessage(
        available=True,
        message="Sample predictions generated successfully.",
        details={
            **_extract_metadata(selected_model_row),
            **feature_details,
            "prediction_row_count": int(len(merged_df)),
            "resolved_feature_count": int(len(feature_columns)),
        },
    )


__all__ = ["build_sample_predictions"]
