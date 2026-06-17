"""Run error analysis for the selected final models."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.split import reconstruct_split_frames
from src.evaluation.error_analysis import build_error_analysis_frame, build_error_summary, select_representative_cases
from src.evaluation.metrics import compute_classification_metrics
from src.features.metrics_features import build_metrics_training_frame
from src.models.predict import load_model, predict_with_model
from src.utils.coercion import coerce_bool, coerce_float
from src.utils.io import read_csv, read_parquet, write_csv, write_json
from src.utils.logging import get_logger
from src.utils.paths import PROCESSED_DATA_DIR, RESULTS_FIGURES_DIR, RESULTS_TABLES_DIR, SPLITS_DIR, ensure_project_dirs

SHAP_FIGURES_DIR = RESULTS_FIGURES_DIR / "shap"
SHAP_TOP_K = 3


def _load_shap_top_features(dataset_name: str, top_k: int = SHAP_TOP_K) -> list[str]:
    summary_path = SHAP_FIGURES_DIR / dataset_name / f"{dataset_name}_shap_global_summary.csv"
    if not summary_path.exists():
        return []
    try:
        summary = read_csv(summary_path)
    except Exception:
        return []
    if summary.empty or "feature" not in summary.columns:
        return []
    metric_column = "mean_abs_shap" if "mean_abs_shap" in summary.columns else ("importance" if "importance" in summary.columns else None)
    if metric_column is None:
        return []
    ranked = summary.sort_values(metric_column, ascending=False)
    return ranked["feature"].astype(str).head(top_k).tolist()


def _attach_top_shap_features(frame, top_features):
    if frame is None or frame.empty or not top_features:
        return frame
    annotated = frame.copy()
    for index, feature in enumerate(top_features, start=1):
        annotated[f"top_shap_feature_{index}"] = feature
    annotated["top_shap_features"] = ",".join(top_features)
    return annotated
from src.utils.provenance import artifact_uses_commit_text

logger = get_logger(__name__)
FINAL_MODELS_PATH = RESULTS_TABLES_DIR / "final_models_by_dataset.csv"
BEST_MODELS_PATH = RESULTS_TABLES_DIR / "best_models_by_dataset.csv"
ERROR_ANALYSIS_SUMMARY_PATH = RESULTS_TABLES_DIR / "error_analysis_summary.csv"
ERROR_ANALYSIS_CASES_PATH = RESULTS_TABLES_DIR / "error_analysis_cases.csv"
ERROR_ANALYSIS_FAILURES_PATH = RESULTS_TABLES_DIR / "error_analysis_failures.csv"
ERROR_ANALYSIS_REPRESENTATIVE_PATH = RESULTS_TABLES_DIR / "error_analysis_representative_cases.csv"
ERROR_ANALYSIS_META_PATH = RESULTS_TABLES_DIR / "error_analysis_meta.json"
HYBRID_FEATURE_FAMILIES = {"metrics_plus_commit_text", "metrics_plus_text", "hybrid"}
DEFAULT_METRICS = ["loc", "v(g)", "ev(g)", "iv(g)", "branchCount", "coupling", "cohesion", "code_churn"]
ERROR_ANALYSIS_FAILURE_COLUMNS = [
    "dataset_name",
    "model",
    "model_path",
    "feature_family",
    "feature_set",
    "text_feature_column",
    "uses_commit_text",
    "artifact_schema_version",
    "artifact_stage",
    "artifact_id",
    "artifact_group_key",
    "source_results_table",
    "error",
]


def _normalize_feature_family(row: pd.Series) -> str:
    for key in ("feature_family", "feature_set"):
        value = row.get(key)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value)
    return "metrics_only"


def build_feature_frame(df: pd.DataFrame, metrics: list[str], selection_context: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the feature frame matching the selected model family."""
    feature_family = str(selection_context.get("feature_family") or selection_context.get("feature_set") or "metrics_only")
    if feature_family in HYBRID_FEATURE_FAMILIES or artifact_uses_commit_text(selection_context):
        raise ValueError(
            "Legacy commit-text artifact cannot be reconstructed for error analysis without a train-fitted ModelBundle."
        )

    feature_frame, _, feature_metadata = build_metrics_training_frame(df, metrics)
    feature_metadata.setdefault("feature_family", "metrics_only")
    return feature_frame, feature_metadata


def _resolve_metrics(df: pd.DataFrame) -> list[str]:
    return [metric for metric in DEFAULT_METRICS if metric in df.columns]


def _selection_context_from_row(row: pd.Series, dataset_name: str, model_name: str) -> dict[str, Any]:
    feature_family = _normalize_feature_family(row)
    text_feature_column = row.get("text_feature_column", "")
    uses_commit_text = artifact_uses_commit_text(row.to_dict())
    selection_context = {
        "training_mode": row.get("training_mode", ""),
        "selection_rank": row.get("selection_rank", 1),
        "selected_reason": row.get("selected_reason", ""),
        "feature_family": feature_family,
        "feature_set": row.get("feature_set", feature_family),
        "text_feature_column": text_feature_column,
        "uses_commit_text": uses_commit_text,
        "artifact_schema_version": row.get("artifact_schema_version", "paper-v1"),
        "artifact_id": row.get("artifact_id", f"{dataset_name}::{model_name}::error_analysis"),
        "artifact_group_key": row.get("artifact_group_key", f"{dataset_name}::{model_name}"),
        "artifact_created_at": row.get("artifact_created_at", ""),
        "source_results_table": row.get("source_results_table", ""),
        "model_path": row.get("model_path", ""),
        "decision_threshold": coerce_float(row.get("decision_threshold"), default=None),
        "split_manifest_path": row.get("split_manifest_path", ""),
        "split_mode": row.get("split_mode", "saved_split"),
    }
    return selection_context


def _build_error_summary_record(analysis_df: pd.DataFrame, dataset_name: str, model_name: str, selection_context: dict[str, Any], feature_metadata: dict[str, Any]) -> dict[str, Any]:
    summary = build_error_summary(analysis_df, dataset_name=dataset_name, model_name=model_name)
    probability = analysis_df["probability"] if "probability" in analysis_df.columns else None
    summary.update(compute_classification_metrics(analysis_df["label"], analysis_df["prediction"], probability))
    summary.update({
        "feature_family": selection_context.get("feature_family", "metrics_only"),
        "feature_set": selection_context.get("feature_set", selection_context.get("feature_family", "metrics_only")),
        "text_feature_column": selection_context.get("text_feature_column", ""),
        "uses_commit_text": coerce_bool(selection_context.get("uses_commit_text", False)),
        "artifact_schema_version": selection_context.get("artifact_schema_version", "paper-v1"),
        "artifact_id": selection_context.get("artifact_id", f"{dataset_name}::{model_name}::error_analysis"),
        "artifact_group_key": selection_context.get("artifact_group_key", f"{dataset_name}::{model_name}"),
        "artifact_stage": "error_analysis",
        "artifact_created_at": selection_context.get("artifact_created_at", ""),
        "source_results_table": selection_context.get("source_results_table", ""),
        "model_path": selection_context.get("model_path", ""),
        "decision_threshold": selection_context.get("decision_threshold"),
        "split_mode": selection_context.get("split_mode", "saved_split"),
        "split_manifest_path": selection_context.get("split_manifest_path", ""),
        "num_features": int(feature_metadata.get("num_features", 0)),
    })
    return summary


def load_final_models_table() -> pd.DataFrame:
    """Load the final-model table created by evaluation."""
    if FINAL_MODELS_PATH.exists():
        return read_csv(FINAL_MODELS_PATH)
    if BEST_MODELS_PATH.exists():
        return read_csv(BEST_MODELS_PATH)
    raise FileNotFoundError(f"Missing final-model summary: {FINAL_MODELS_PATH}")


def load_clean_dataset(dataset_name: str) -> pd.DataFrame:
    """Load the cleaned baseline dataset for one project."""
    dataset_path = PROCESSED_DATA_DIR / f"{dataset_name}_clean.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing cleaned dataset: {dataset_path}")
    return read_parquet(dataset_path)


JITLINE_DATASETS = {"openstack", "qt", "jitfine"}

def load_test_dataset(dataset_name: str, split_mode: str | None = None) -> pd.DataFrame:
    """Load the saved test split for one dataset."""
    df = load_clean_dataset(dataset_name)
    mode = (split_mode or "saved_split").strip().lower()
    if dataset_name in JITLINE_DATASETS or mode == "jitline_native_split":
        if "jitline_split" not in df.columns:
            raise ValueError(f"Dataset {dataset_name} is missing the jitline_split column required for the native split.")
        normalized = df["jitline_split"].astype(str).str.strip().str.lower()
        test_df = df.loc[normalized == "test"].copy()
        if test_df.empty:
            raise ValueError(f"JITLine native split for {dataset_name} produced an empty test frame.")
        return test_df
    split_dir = SPLITS_DIR / dataset_name
    _, _, test_df = reconstruct_split_frames(
        df,
        split_dir / "train_ids.csv",
        split_dir / "val_ids.csv",
        split_dir / "test_ids.csv",
    )
    return test_df


def build_case_frame(dataset_name: str, model_name: str, model_path: str, df: pd.DataFrame, metrics: list[str], selection_context: dict[str, Any]) -> pd.DataFrame:
    """Build a row-level error analysis frame for one dataset/model pair."""
    model = load_model(model_path)

    if "label" not in df.columns:
        raise ValueError(f"Dataset {dataset_name} is missing the label column")

    if hasattr(model, "transform_features"):
        feature_frame = model.transform_features(df)
        feature_metadata = {
            "feature_family": getattr(model, "feature_family", selection_context.get("feature_family", "metrics_only")),
            "num_features": int(feature_frame.shape[1]),
        }
    else:
        feature_frame, feature_metadata = build_feature_frame(df, metrics, selection_context)
    if feature_frame.empty or feature_frame.shape[1] == 0:
        raise ValueError(f"No usable features found for dataset {dataset_name} and model {model_name}")

    decision_threshold = selection_context.get("decision_threshold")
    predictions = predict_with_model(
        model,
        df if hasattr(model, "transform_features") else feature_frame,
        threshold=decision_threshold,
    )

    analysis_df = df.copy().reset_index(drop=True)
    analysis_df = pd.concat([analysis_df, predictions.reset_index(drop=True)], axis=1)
    analysis_df = build_error_analysis_frame(analysis_df, label_col="label", pred_col="prediction")
    for column_name in ("dataset_name", "model"):
        if column_name in analysis_df.columns:
            analysis_df = analysis_df.drop(columns=[column_name])
    analysis_df.insert(0, "dataset_name", dataset_name)
    analysis_df.insert(1, "model", model_name)
    analysis_df["feature_family"] = selection_context.get("feature_family", feature_metadata.get("feature_family", "metrics_only"))
    analysis_df["feature_set"] = selection_context.get("feature_set", analysis_df["feature_family"])
    analysis_df["text_feature_column"] = selection_context.get("text_feature_column", "")
    analysis_df["uses_commit_text"] = coerce_bool(selection_context.get("uses_commit_text", False))
    analysis_df["artifact_schema_version"] = selection_context.get("artifact_schema_version", "paper-v1")
    analysis_df["artifact_stage"] = "error_analysis"
    analysis_df["artifact_id"] = selection_context.get("artifact_id", f"{dataset_name}::{model_name}::error_analysis")
    analysis_df["artifact_group_key"] = selection_context.get("artifact_group_key", f"{dataset_name}::{model_name}")
    analysis_df["artifact_created_at"] = selection_context.get("artifact_created_at", "")
    analysis_df["source_results_table"] = selection_context.get("source_results_table", "")
    analysis_df["model_path"] = selection_context.get("model_path", "")
    analysis_df["decision_threshold"] = selection_context.get("decision_threshold")
    analysis_df["split_mode"] = selection_context.get("split_mode", "saved_split")
    analysis_df["split_manifest_path"] = selection_context.get("split_manifest_path", "")
    analysis_df["feature_metadata_json"] = str(feature_metadata)
    analysis_df["num_features"] = int(feature_metadata.get("num_features", feature_frame.shape[1]))
    for key, value in selection_context.items():
        if key not in analysis_df.columns:
            analysis_df[key] = value
    return analysis_df


def main() -> None:
    """Run error analysis for every selected final model."""
    ensure_project_dirs()
    best_models_df = load_final_models_table()
    logger.info("Loaded %s final-model row(s) for error analysis.", len(best_models_df))

    all_cases: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []
    representative_frames: list[pd.DataFrame] = []

    for _, row in best_models_df.iterrows():
        dataset_name = str(row["dataset_name"])
        model_name = str(row["model"])
        model_path = str(row["model_path"])
        selection_context = _selection_context_from_row(row, dataset_name, model_name)

        logger.info("Running error analysis for dataset=%s model=%s", dataset_name, model_name)
        try:
            df = load_test_dataset(dataset_name, split_mode=str(selection_context.get("split_mode", "saved_split")))
            metrics = _resolve_metrics(df)
            analysis_df = build_case_frame(dataset_name, model_name, model_path, df, metrics, selection_context)
            feature_metadata = {"num_features": int(analysis_df["num_features"].iloc[0]) if not analysis_df.empty and "num_features" in analysis_df.columns else 0}
            summary_records.append(_build_error_summary_record(analysis_df, dataset_name=dataset_name, model_name=model_name, selection_context=selection_context, feature_metadata=feature_metadata))
            all_cases.append(analysis_df)
            top_shap_features = _load_shap_top_features(dataset_name)
            representative_df = select_representative_cases(analysis_df)
            representative_df = _attach_top_shap_features(representative_df, top_shap_features)
            representative_frames.append(representative_df)
        except Exception as exc:
            logger.exception("Error analysis failed for dataset=%s model=%s: %s", dataset_name, model_name, exc)
            failure_records.append({
                "dataset_name": dataset_name,
                "model": model_name,
                "model_path": model_path,
                "feature_family": selection_context.get("feature_family", "metrics_only"),
                "feature_set": selection_context.get("feature_set", selection_context.get("feature_family", "metrics_only")),
                "text_feature_column": selection_context.get("text_feature_column", ""),
                "uses_commit_text": coerce_bool(selection_context.get("uses_commit_text", False)),
                "artifact_schema_version": selection_context.get("artifact_schema_version", "paper-v1"),
                "artifact_stage": "error_analysis",
                "artifact_id": selection_context.get("artifact_id", f"{dataset_name}::{model_name}::error_analysis"),
                "artifact_group_key": selection_context.get("artifact_group_key", f"{dataset_name}::{model_name}"),
                "source_results_table": selection_context.get("source_results_table", ""),
                "error": str(exc),
            })

    cases_df = pd.concat(all_cases, ignore_index=True) if all_cases else pd.DataFrame()
    summary_df = pd.DataFrame(summary_records)
    failures_df = pd.DataFrame(failure_records)
    if failures_df.empty and len(failures_df.columns) == 0:
        failures_df = pd.DataFrame(columns=ERROR_ANALYSIS_FAILURE_COLUMNS)
    representative_df = pd.concat(representative_frames, ignore_index=True) if representative_frames else pd.DataFrame()

    write_csv(summary_df, ERROR_ANALYSIS_SUMMARY_PATH)
    write_csv(cases_df, ERROR_ANALYSIS_CASES_PATH)
    write_csv(failures_df, ERROR_ANALYSIS_FAILURES_PATH)
    write_csv(representative_df, ERROR_ANALYSIS_REPRESENTATIVE_PATH)
    write_json(
        {
            "source_final_models": str(FINAL_MODELS_PATH if FINAL_MODELS_PATH.exists() else BEST_MODELS_PATH),
            "error_analysis_summary": str(ERROR_ANALYSIS_SUMMARY_PATH),
            "error_analysis_cases": str(ERROR_ANALYSIS_CASES_PATH),
            "error_analysis_failures": str(ERROR_ANALYSIS_FAILURES_PATH),
            "error_analysis_representative_cases": str(ERROR_ANALYSIS_REPRESENTATIVE_PATH),
            "num_case_rows": int(len(cases_df)),
            "num_summary_rows": int(len(summary_df)),
            "num_failure_rows": int(len(failures_df)),
            "artifact_schema_version": "paper-v1",
        },
        ERROR_ANALYSIS_META_PATH,
    )
    logger.info("Saved error analysis summary to %s", ERROR_ANALYSIS_SUMMARY_PATH)
    logger.info("Saved error analysis cases to %s", ERROR_ANALYSIS_CASES_PATH)
    logger.info("Saved error analysis failures to %s", ERROR_ANALYSIS_FAILURES_PATH)
    logger.info("Saved error analysis representative cases to %s", ERROR_ANALYSIS_REPRESENTATIVE_PATH)
    logger.info("Saved error analysis metadata to %s", ERROR_ANALYSIS_META_PATH)


if __name__ == "__main__":
    main()
