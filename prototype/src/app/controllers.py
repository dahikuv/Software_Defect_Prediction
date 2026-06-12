"""Controller helpers for the Streamlit MVC app."""

from __future__ import annotations

import pandas as pd

from src.app.services.dataset_service import build_sample_rows
from src.app.services.evaluation_service import (
    describe_model_selection,
    extract_metric_summary,
    get_model_artifact_metadata,
    list_available_datasets,
    list_available_models_for_dataset,
    load_best_models_table,
    load_model_ranking,
    row_to_dict,
)
from src.app.services.explainability_service import (
    build_explainability_artifacts,
    build_explainability_status,
    load_model_artifact_previews,
    load_model_scoped_artifact_rows,
)
from src.app.services.model_service import build_sample_predictions
from src.app.state import DatasetDashboardState, ExplainabilityArtifacts

HYBRID_FEATURE_FAMILIES = {"metrics_plus_commit_text", "metrics_plus_text", "hybrid"}


def _resolve_feature_family(selected_model_row: dict[str, str], metric_summary: dict[str, str], dataset_status) -> str:
    feature_family = (
        selected_model_row.get("feature_family")
        or selected_model_row.get("feature_set")
        or metric_summary.get("feature_family")
        or metric_summary.get("feature_set")
    )
    if feature_family:
        return str(feature_family)
    return "metrics_plus_commit_text" if dataset_status.details.get("commit_text_available") or dataset_status.details.get("has_commit_text") else "metrics_only"


def _resolve_explainability_mode(explainability_status) -> str:
    if explainability_status.available and explainability_status.details.get("model_scoped"):
        return "artifact"
    if explainability_status.available:
        return "partial"
    return "fallback"


def _build_dashboard_notes(feature_family: str, commit_text_column: str | None, metrics_for_ui: dict[str, str], explainability_status, configured_hybrid: bool, commit_text_available: bool) -> list[str]:
    notes = [
        "Use the paper-facing dataset and model views for the research flow.",
        "Use live repo analysis only as heuristic fallback when needed.",
    ]
    if explainability_status.available and not explainability_status.details.get("model_scoped"):
        notes.append("Explainability previews are available, but model-specific linkage is incomplete; dataset-level previews are shown.")
    if commit_text_column:
        notes.append(f"Commit-text feature column: {commit_text_column}")
    if feature_family == "metrics_plus_commit_text":
        notes.append("This model/view is aligned with the paper's metrics + commit-messages direction.")
    else:
        notes.append("This model/view currently behaves like a metrics-first baseline.")
    if metrics_for_ui.get("num_features"):
        notes.append(f"Artifact summary reports {metrics_for_ui['num_features']} input features.")
    if metrics_for_ui.get("auc") is not None:
        notes.append(f"Artifact AUC: {metrics_for_ui['auc']}")
    if configured_hybrid and not commit_text_available:
        notes.append("Commit-text artifacts are configured, but the current sample rows do not expose commit text.")
    return notes


def build_dashboard_state(dataset_name: str, selected_model: str | None = None) -> DatasetDashboardState:
    """Build the full UI state for one dataset selection."""
    best_df = load_best_models_table()
    ranking_df = load_model_ranking()

    best_row_df = best_df[best_df["dataset_name"] == dataset_name] if not best_df.empty else pd.DataFrame()
    best_row = best_row_df.iloc[0] if not best_row_df.empty else None
    best_model = str(best_row["model"]) if best_row is not None and "model" in best_row.index else ""

    model_options = list_available_models_for_dataset(dataset_name)
    resolved_model = selected_model or best_model or (model_options[0] if model_options else "")

    selected_row_df = ranking_df[
        (ranking_df["dataset_name"] == dataset_name) & (ranking_df["model"] == resolved_model)
    ] if not ranking_df.empty else pd.DataFrame()
    if selected_row_df.empty and not best_row_df.empty and resolved_model == best_model:
        selected_row_df = best_row_df.copy()
    selected_row = selected_row_df.iloc[0] if not selected_row_df.empty else None

    ranking_rows_df = ranking_df[ranking_df["dataset_name"] == dataset_name].copy() if not ranking_df.empty else pd.DataFrame()
    if not ranking_rows_df.empty and "rank_within_dataset" in ranking_rows_df.columns:
        ranking_rows_df = ranking_rows_df.sort_values("rank_within_dataset")

    best_model_row = row_to_dict(best_row)
    selected_model_row = row_to_dict(selected_row)
    artifact_metadata = get_model_artifact_metadata(dataset_name, resolved_model)
    best_model_row = {**artifact_metadata, **best_model_row}
    selected_model_row = {**artifact_metadata, **selected_model_row}
    if best_model_row.get("feature_family") and not best_model_row.get("feature_set"):
        best_model_row["feature_set"] = best_model_row["feature_family"]
    if selected_model_row.get("feature_family") and not selected_model_row.get("feature_set"):
        selected_model_row["feature_set"] = selected_model_row["feature_family"]
    if selected_model_row.get("text_feature_column") and not selected_model_row.get("commit_text_column"):
        selected_model_row["commit_text_column"] = selected_model_row["text_feature_column"]
    if best_model_row.get("text_feature_column") and not best_model_row.get("commit_text_column"):
        best_model_row["commit_text_column"] = best_model_row["text_feature_column"]
    selected_model_row["commit_text_available"] = bool(selected_model_row.get("commit_text_available") or selected_model_row.get("uses_commit_text") or selected_model_row.get("text_feature_column"))
    best_model_row["commit_text_available"] = bool(best_model_row.get("commit_text_available") or best_model_row.get("uses_commit_text") or best_model_row.get("text_feature_column"))
    selected_model_row["uses_commit_text"] = bool(selected_model_row.get("uses_commit_text") or selected_model_row.get("commit_text_available"))
    best_model_row["uses_commit_text"] = bool(best_model_row.get("uses_commit_text") or best_model_row.get("commit_text_available"))

    explainability = build_explainability_artifacts(dataset_name, resolved_model)
    previews = load_model_artifact_previews(dataset_name, resolved_model)
    sample_rows, metric_columns, sample_df, dataset_status = build_sample_rows(dataset_name)
    metric_summary = extract_metric_summary(dataset_name, resolved_model)

    commit_text_column = (
        selected_model_row.get("text_feature_column")
        or metric_summary.get("text_feature_column")
        or dataset_status.details.get("commit_text_column")
    )
    if commit_text_column:
        dataset_status.details["commit_text_column"] = commit_text_column
        selected_model_row["text_feature_column"] = commit_text_column
        best_model_row.setdefault("text_feature_column", commit_text_column)

    feature_family = _resolve_feature_family(selected_model_row, metric_summary, dataset_status)
    commit_text_available = bool(
        dataset_status.details.get("commit_text_available", dataset_status.details.get("has_commit_text"))
        or selected_model_row.get("commit_text_available")
        or selected_model_row.get("uses_commit_text")
        or selected_model_row.get("text_feature_column")
        or best_model_row.get("commit_text_available")
        or best_model_row.get("uses_commit_text")
        or best_model_row.get("text_feature_column")
    )
    dataset_status.details["commit_text_available"] = commit_text_available
    dataset_status.details["has_commit_text"] = commit_text_available
    dataset_status.details.setdefault("commit_text_column", selected_model_row.get("commit_text_column") or best_model_row.get("commit_text_column"))
    paper_metric_columns = list(dataset_status.details.get("paper_metric_columns", metric_columns))
    dataset_status.details["commit_text_available"] = commit_text_available
    dataset_status.details["paper_metric_columns"] = paper_metric_columns
    configured_hybrid = feature_family in HYBRID_FEATURE_FAMILIES
    if configured_hybrid and not commit_text_available:
        feature_family = "metrics_only"

    selected_model_row["feature_family"] = feature_family
    selected_model_row.setdefault("feature_set", feature_family)
    selected_model_row["commit_text_available"] = commit_text_available
    selected_model_row["uses_commit_text"] = bool(selected_model_row.get("uses_commit_text") or commit_text_available)
    selected_model_row["paper_metric_columns"] = paper_metric_columns
    best_model_row["feature_family"] = best_model_row.get("feature_family") or feature_family
    best_model_row.setdefault("feature_set", best_model_row["feature_family"])
    best_model_row["paper_metric_columns"] = paper_metric_columns
    best_model_row["commit_text_available"] = commit_text_available
    best_model_row["uses_commit_text"] = bool(best_model_row.get("uses_commit_text") or commit_text_available)
    selected_model_row["artifact_schema_version"] = selected_model_row.get("artifact_schema_version") or metric_summary.get("artifact_schema_version")
    selected_model_row["artifact_stage"] = selected_model_row.get("artifact_stage") or metric_summary.get("artifact_stage")
    selected_model_row["artifact_group_key"] = selected_model_row.get("artifact_group_key") or metric_summary.get("artifact_group_key")
    selected_model_row["artifact_id"] = selected_model_row.get("artifact_id") or metric_summary.get("artifact_id")
    selected_model_row["source_results_table"] = selected_model_row.get("source_results_table") or metric_summary.get("source_results_table")

    metrics_for_ui = {**metric_summary, "feature_family": feature_family}
    if commit_text_column:
        metrics_for_ui["text_feature_column"] = commit_text_column
    if paper_metric_columns:
        metrics_for_ui["paper_metric_columns"] = paper_metric_columns
    metrics_for_ui["has_commit_text"] = commit_text_available

    model_status = describe_model_selection(dataset_name, resolved_model, selected_model_row)
    sample_prediction_rows, prediction_status = build_sample_predictions(selected_model_row, sample_df, metric_columns)
    prediction_status.details.setdefault("feature_family", feature_family)
    prediction_status.details["has_commit_text"] = commit_text_available
    prediction_status.details["paper_metric_columns"] = paper_metric_columns
    if commit_text_column:
        prediction_status.details.setdefault("text_feature_column", commit_text_column)
    if configured_hybrid and not commit_text_available:
        prediction_status.details["feature_family_fallback"] = "metrics_only"
        prediction_status.details["feature_family_reason"] = "commit text is not available in the current sample rows"

    explainability_status = build_explainability_status(
        dataset_name=dataset_name,
        selected_model=resolved_model,
        artifacts=explainability,
        shap_rows=previews.shap_rows,
        impact_rows=previews.impact_rows,
        error_summary_rows=previews.error_summary_rows,
    )

    explanation_mode = _resolve_explainability_mode(explainability_status)

    notes = _build_dashboard_notes(
        feature_family=feature_family,
        commit_text_column=commit_text_column,
        metrics_for_ui=metrics_for_ui,
        explainability_status=explainability_status,
        configured_hybrid=configured_hybrid,
        commit_text_available=commit_text_available,
    )
    if explanation_mode == "partial" and not explainability_status.details.get("model_scoped"):
        notes.append("Explainability mode is partial because backend artifacts exist but are not fully model-scoped.")
    if explanation_mode == "fallback" and not explainability_status.available:
        notes.append("Explainability artifacts are unavailable for this selection; fallback notes are shown instead.")
    if commit_text_available and not commit_text_column:
        notes.append("Commit text is present in the dataset, but no specific text feature column was selected.")

    return DatasetDashboardState(
        dataset_name=dataset_name,
        selected_model=resolved_model,
        best_model=best_model,
        model_options=model_options,
        metrics=metrics_for_ui,
        ranking_rows=ranking_rows_df.to_dict("records") if not ranking_rows_df.empty else [],
        best_model_row=best_model_row,
        selected_model_row=selected_model_row,
        explainability=ExplainabilityArtifacts(
            global_summary_csv=explainability.global_summary_csv,
            importance_csv=explainability.importance_csv,
            summary_plot=explainability.summary_plot,
            local_csv=explainability.local_csv,
            status=explainability_status,
        ),
        sample_rows=sample_rows,
        sample_prediction_rows=sample_prediction_rows,
        sample_metrics=metric_columns,
        global_explainability_rows=previews.shap_rows,
        shap_local_rows=load_model_scoped_artifact_rows(explainability.local_csv, resolved_model),
        dataset_status=dataset_status,
        model_status=model_status,
        prediction_status=prediction_status,
        feature_family=feature_family,
        commit_text_available=commit_text_available,
        paper_metric_columns=paper_metric_columns,
        explanation_mode=explanation_mode,
        notes=notes,
        impact_rows=previews.impact_rows,
        error_summary_rows=previews.error_summary_rows,
        error_case_rows=previews.error_case_rows,
    )


__all__ = [
    "DatasetDashboardState",
    "build_dashboard_state",
    "list_available_datasets",
    "list_available_models_for_dataset",
    "load_best_models_table",
    "load_model_ranking",
]
