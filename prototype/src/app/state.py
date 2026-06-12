"""Shared app state models for the MVC backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StatusMessage:
    """Availability and error metadata for one backend resource."""

    available: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplainabilityArtifacts:
    """Paths to explainability artifacts for one dataset."""

    global_summary_csv: str | None
    importance_csv: str | None
    summary_plot: str | None
    local_csv: str | None
    status: StatusMessage


@dataclass
class DatasetDashboardState:
    """UI-ready state for one dataset/model selection."""

    dataset_name: str
    selected_model: str
    best_model: str
    model_options: list[str]
    metrics: dict[str, Any]
    ranking_rows: list[dict[str, Any]]
    best_model_row: dict[str, Any]
    selected_model_row: dict[str, Any]
    explainability: ExplainabilityArtifacts
    sample_rows: list[dict[str, Any]]
    sample_prediction_rows: list[dict[str, Any]]
    sample_metrics: list[str]
    global_explainability_rows: list[dict[str, Any]]
    shap_local_rows: list[dict[str, Any]]
    dataset_status: StatusMessage
    model_status: StatusMessage
    prediction_status: StatusMessage
    feature_family: str = "metrics_only"
    commit_text_available: bool = False
    paper_metric_columns: list[str] = field(default_factory=list)
    impact_rows: list[dict[str, Any]] = field(default_factory=list)
    error_summary_rows: list[dict[str, Any]] = field(default_factory=list)
    error_case_rows: list[dict[str, Any]] = field(default_factory=list)
    explanation_mode: str = "artifact"
    notes: list[str] = field(default_factory=list)


@dataclass
class AnalysisResultRow:
    """A single analyzed file row with reasoning metadata."""

    path: str
    probability: str
    severity: str
    reason: str
    signals: list[str]
    source_type: str = "heuristic"
    model_probability: str | None = None
    model_prediction: Any | None = None
    skipped: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class AnalysisResultState:
    """UI-ready analysis result for the live product flow."""

    source: str
    file_count: int
    risks: list[AnalysisResultRow]
    notes: list[str] = field(default_factory=list)
    excluded_files: list[str] = field(default_factory=list)
    explainability: StatusMessage | None = None


__all__ = ["AnalysisResultState", "AnalysisResultRow", "DatasetDashboardState", "ExplainabilityArtifacts", "StatusMessage"]
