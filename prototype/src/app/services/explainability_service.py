"""Explainability artifact helpers for the MVC backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.app.state import ExplainabilityArtifacts, StatusMessage
from src.utils.io import read_csv
from src.utils.paths import RESULTS_FIGURES_DIR, RESULTS_TABLES_DIR

SHAP_DIR = RESULTS_FIGURES_DIR / "shap"
SHAP_SUMMARY_TABLE_PATH = RESULTS_TABLES_DIR / "shap_explainability_summary.csv"
ERROR_ANALYSIS_SUMMARY_PATH = RESULTS_TABLES_DIR / "error_analysis_summary.csv"
ERROR_ANALYSIS_REPRESENTATIVE_PATH = RESULTS_TABLES_DIR / "error_analysis_representative_cases.csv"
IMPACT_SUMMARY_PATH = RESULTS_TABLES_DIR / "commit_message_impact_summary.csv"
IMPACT_TABLE_PATH = RESULTS_TABLES_DIR / "commit_message_impact.csv"


@dataclass
class ArtifactPreviewBundle:
    """Grouped preview rows for the dashboard."""

    shap_rows: list[dict[str, Any]]
    error_summary_rows: list[dict[str, Any]]
    error_case_rows: list[dict[str, Any]]
    impact_rows: list[dict[str, Any]]


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_csv(path)


def build_explainability_artifacts(dataset_name: str, selected_model: str | None = None) -> ExplainabilityArtifacts:
    """Resolve explainability artifact paths for one dataset/model selection."""
    dataset_dir = SHAP_DIR / dataset_name

    def maybe(path: Path) -> str | None:
        return str(path) if path.exists() else None

    global_summary_csv = maybe(dataset_dir / f"{dataset_name}_shap_global_summary.csv")
    importance_csv = maybe(dataset_dir / f"{dataset_name}_shap_importance.csv")
    summary_plot = maybe(dataset_dir / f"{dataset_name}_shap_summary.png")
    local_csv = maybe(dataset_dir / f"{dataset_name}_test_row_0_shap_local.csv")
    available_paths = [path for path in [global_summary_csv, importance_csv, summary_plot, local_csv] if path]

    status = StatusMessage(
        available=bool(available_paths),
        message="Explainability artifacts are available." if available_paths else "No explainability artifacts were found.",
        details={
            "dataset_name": dataset_name,
            "selected_model": selected_model,
            "artifact_dir": str(dataset_dir),
            "available_count": len(available_paths),
            "model_scoped": False,
            "artifact_scope": "dataset" if available_paths else "none",
        },
    )

    return ExplainabilityArtifacts(
        global_summary_csv=global_summary_csv,
        importance_csv=importance_csv,
        summary_plot=summary_plot,
        local_csv=local_csv,
        status=status,
    )


def _filter_rows_for_model(rows: list[dict[str, Any]], selected_model: str | None) -> list[dict[str, Any]]:
    """Prefer rows matching the selected model when model metadata is available."""
    if not rows or not selected_model:
        return rows
    matching = [row for row in rows if str(row.get("model", "")) == str(selected_model)]
    return matching or rows


def build_explainability_status(
    dataset_name: str,
    selected_model: str | None,
    artifacts: ExplainabilityArtifacts,
    shap_rows: list[dict[str, Any]],
    impact_rows: list[dict[str, Any]],
    error_summary_rows: list[dict[str, Any]],
) -> StatusMessage:
    """Build a richer explainability status for dashboard verification."""
    selected_model_str = str(selected_model) if selected_model else ""
    model_scoped = any(str(row.get("model", "")) == selected_model_str for row in [*shap_rows, *impact_rows, *error_summary_rows]) if selected_model else False
    available = artifacts.status.available or bool(shap_rows or impact_rows or error_summary_rows)
    message = "Explainability artifacts are available."
    if available and selected_model and not model_scoped:
        message = "Explainability artifacts exist, but model-specific linkage is incomplete; showing dataset-level previews."
    elif not available:
        message = "No explainability artifacts were found."

    return StatusMessage(
        available=available,
        message=message,
        details={
            "dataset_name": dataset_name,
            "selected_model": selected_model,
            "model_scoped": model_scoped,
            "artifact_scope": "model" if model_scoped else (artifacts.status.details.get("artifact_scope") or "dataset"),
            "shap_row_count": len(shap_rows),
            "impact_row_count": len(impact_rows),
            "error_summary_row_count": len(error_summary_rows),
            "available_count": artifacts.status.details.get("available_count", 0),
        },
    )


def load_model_artifact_previews(dataset_name: str, selected_model: str | None) -> ArtifactPreviewBundle:
    """Load dashboard previews and prefer rows matching the selected model."""
    previews = load_dashboard_artifact_previews(dataset_name)
    return ArtifactPreviewBundle(
        shap_rows=_filter_rows_for_model(previews.shap_rows, selected_model),
        error_summary_rows=_filter_rows_for_model(previews.error_summary_rows, selected_model),
        error_case_rows=_filter_rows_for_model(previews.error_case_rows, selected_model),
        impact_rows=_filter_rows_for_model(previews.impact_rows, selected_model),
    )


def load_model_scoped_artifact_rows(path: str | None, selected_model: str | None, limit: int = 10) -> list[dict[str, Any]]:
    """Load artifact rows and prefer model-matching records when possible."""
    rows = load_artifact_rows(path, limit=limit)
    return _filter_rows_for_model(rows, selected_model)


def load_artifact_rows(path: str | None, limit: int = 10) -> list[dict[str, Any]]:
    """Load preview rows from an artifact CSV."""
    if not path:
        return []

    try:
        artifact_df = read_csv(path)
    except Exception:
        return []

    if artifact_df.empty:
        return []
    return artifact_df.head(limit).to_dict("records")


def load_dashboard_artifact_previews(dataset_name: str) -> ArtifactPreviewBundle:
    """Load dashboard previews for explainability, impact, and error analysis."""
    shap_df = _read_optional_csv(SHAP_SUMMARY_TABLE_PATH)
    error_summary_df = _read_optional_csv(ERROR_ANALYSIS_SUMMARY_PATH)
    error_case_df = _read_optional_csv(ERROR_ANALYSIS_REPRESENTATIVE_PATH)
    impact_df = _read_optional_csv(IMPACT_SUMMARY_PATH)

    shap_rows = []
    if not shap_df.empty and "dataset_name" in shap_df.columns:
        shap_rows = shap_df[shap_df["dataset_name"] == dataset_name].head(10).to_dict("records")

    error_summary_rows = []
    if not error_summary_df.empty and "dataset_name" in error_summary_df.columns:
        error_summary_rows = error_summary_df[error_summary_df["dataset_name"] == dataset_name].head(10).to_dict("records")

    error_case_rows = []
    if not error_case_df.empty and "dataset_name" in error_case_df.columns:
        error_case_rows = error_case_df[error_case_df["dataset_name"] == dataset_name].head(10).to_dict("records")

    impact_rows = []
    if not impact_df.empty and "dataset_name" in impact_df.columns:
        impact_rows = impact_df[impact_df["dataset_name"] == dataset_name].head(10).to_dict("records")

    return ArtifactPreviewBundle(
        shap_rows=shap_rows,
        error_summary_rows=error_summary_rows,
        error_case_rows=error_case_rows,
        impact_rows=impact_rows,
    )


__all__ = [
    "ArtifactPreviewBundle",
    "build_explainability_artifacts",
    "build_explainability_status",
    "load_artifact_rows",
    "load_dashboard_artifact_previews",
    "load_model_artifact_previews",
    "load_model_scoped_artifact_rows",
]
