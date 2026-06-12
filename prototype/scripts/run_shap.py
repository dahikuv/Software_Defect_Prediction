"""Run the SHAP scaffold."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.explainability.shap_global import run_global_shap
from src.explainability.shap_local import run_local_shap
from src.features.metrics_features import build_hybrid_training_frame, build_metrics_training_frame
from src.utils.io import read_csv, read_parquet, write_csv, write_json
from src.utils.logging import get_logger
from src.utils.paths import CONFIG_PATH, PROCESSED_DATA_DIR, RESULTS_FIGURES_DIR, RESULTS_TABLES_DIR, ensure_project_dirs

logger = get_logger(__name__)
FINAL_MODELS_PATH = RESULTS_TABLES_DIR / "final_models_by_dataset.csv"
BEST_MODELS_PATH = RESULTS_TABLES_DIR / "best_models_by_dataset.csv"
SHAP_MANIFEST_PATH = RESULTS_TABLES_DIR / "shap_manifest.json"
SHAP_SUMMARY_PATH = RESULTS_TABLES_DIR / "shap_explainability_summary.csv"
DEFAULT_METRICS = ["loc", "v(g)", "ev(g)", "iv(g)", "branchCount", "coupling", "cohesion", "code_churn"]
HYBRID_FEATURE_FAMILIES = {"metrics_plus_commit_text", "metrics_plus_text", "hybrid"}


def load_final_models_table() -> pd.DataFrame:
    """Load the final-model summary produced by evaluation."""
    if FINAL_MODELS_PATH.exists():
        return read_csv(FINAL_MODELS_PATH)
    if BEST_MODELS_PATH.exists():
        return read_csv(BEST_MODELS_PATH)
    raise FileNotFoundError(f"Missing final-model summary: {FINAL_MODELS_PATH}")


def load_explainability_config() -> dict[str, Any]:
    """Load explainability settings from the project config."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config.get("explainability", {})


def _normalize_feature_family(row: pd.Series) -> str:
    return str(row.get("feature_family") or row.get("feature_set") or "metrics_only")


def load_training_frame_for_dataset(row: pd.Series) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load and rebuild the training frame that matches the selected model family."""
    dataset_name = str(row["dataset_name"])
    dataset_path = PROCESSED_DATA_DIR / f"{dataset_name}_clean.parquet"
    df = read_parquet(dataset_path)
    available_metrics = [metric for metric in DEFAULT_METRICS if metric in df.columns]
    feature_family = _normalize_feature_family(row)

    if feature_family in HYBRID_FEATURE_FAMILIES:
        X, y, metadata = build_hybrid_training_frame(df, available_metrics)
        metadata["resolved_feature_family"] = "metrics_plus_commit_text"
        return X, y, metadata

    X, y, metadata = build_metrics_training_frame(df, available_metrics)
    metadata["resolved_feature_family"] = "metrics_only"
    return X, y, metadata


def _split_feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    metric_columns = [column for column in X.columns if not str(column).startswith("commit_")]
    commit_columns = [column for column in X.columns if str(column).startswith("commit_")]
    return metric_columns, commit_columns


def _sanitize_feature_names(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Convert feature names to SHAP-safe names while preserving a mapping."""
    rename_map = {
        column: str(column)
        .replace("(", "_")
        .replace(")", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        for column in X.columns
    }
    return X.rename(columns=rename_map), rename_map


def _attach_feature_metadata(record: dict[str, Any], feature_metadata: dict[str, Any], X: pd.DataFrame) -> dict[str, Any]:
    metric_columns, commit_columns = _split_feature_columns(X)
    record["metric_feature_columns"] = metric_columns
    record["commit_feature_columns"] = commit_columns
    record["num_features"] = int(X.shape[1])
    record["num_metric_features"] = int(len(metric_columns))
    record["num_commit_features"] = int(len(commit_columns))
    record["resolved_feature_family"] = feature_metadata.get("resolved_feature_family", "metrics_only")
    return record


def _write_feature_manifest(output_dir: Path, dataset_name: str, rename_map: dict[str, str], feature_metadata: dict[str, Any]) -> str:
    manifest_path = output_dir / f"{dataset_name}_shap_feature_manifest.json"
    write_json(
        {
            "dataset_name": dataset_name,
            "feature_name_map": rename_map,
            "feature_metadata": feature_metadata,
        },
        manifest_path,
    )
    return str(manifest_path)


def _build_shap_record(
    row: pd.Series,
    model_path: Path,
    global_outputs: dict[str, str],
    local_outputs: dict[str, str],
    feature_metadata: dict[str, Any],
    X: pd.DataFrame,
    feature_manifest_path: str,
) -> dict[str, Any]:
    dataset_name = str(row["dataset_name"])
    feature_family = _normalize_feature_family(row)
    text_feature_column = str(row.get("text_feature_column", ""))
    uses_commit_text = bool(row.get("uses_commit_text", feature_family in HYBRID_FEATURE_FAMILIES or bool(text_feature_column)))

    record = {
        "dataset_name": dataset_name,
        "model": str(row.get("model", "")),
        "model_path": str(model_path),
        "feature_family": feature_family,
        "feature_set": str(row.get("feature_set", feature_family)),
        "resolved_feature_family": feature_metadata.get("resolved_feature_family", feature_family),
        "text_feature_column": text_feature_column,
        "uses_commit_text": uses_commit_text,
        "artifact_schema_version": str(row.get("artifact_schema_version", "paper-v1")),
        "artifact_stage": "shap",
        "artifact_id": str(row.get("artifact_id", f"{dataset_name}::{row.get('model', '')}::shap")),
        "global_summary_csv": global_outputs.get("summary_csv", ""),
        "global_importance_csv": global_outputs.get("importance_csv", ""),
        "global_plot_path": global_outputs.get("plot_path", ""),
        "local_csv": local_outputs.get("local_csv", ""),
        "feature_manifest_path": feature_manifest_path,
    }
    return _attach_feature_metadata(record, feature_metadata, X)


def log_step(dataset_name: str, message: str) -> None:
    """Log a SHAP progress step for one dataset."""
    logger.info("[%s] %s", dataset_name, message)


def sample_frame(X: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Limit explainability computation size for reproducible runs."""
    if len(X) <= limit:
        return X
    return X.sample(n=limit, random_state=42)


def safe_stratify(y: pd.Series, requested: bool) -> pd.Series | None:
    """Return stratify labels only when safe for splitting."""
    if not requested or y.nunique() <= 1:
        return None
    counts = y.value_counts()
    if counts.empty or counts.min() < 2:
        return None
    return y


def run_dataset_shap(row: pd.Series, explainability_config: dict[str, Any]) -> dict[str, Any]:
    """Run SHAP generation for one selected model row."""
    dataset_name = str(row["dataset_name"])
    model_path = Path(str(row["model_path"]))

    mode = explainability_config.get("mode", "true_shap")
    background_sample_size = int(explainability_config.get("background_sample_size", 100))
    explain_sample_size = int(explainability_config.get("explain_sample_size", 50))
    enable_plots = bool(explainability_config.get("enable_plots", False))
    allow_fallback = bool(explainability_config.get("allow_fallback", True))

    log_step(dataset_name, f"loading training frame from {dataset_name}_clean.parquet")
    X, y, feature_metadata = load_training_frame_for_dataset(row)
    X, rename_map = _sanitize_feature_names(X)
    log_step(dataset_name, f"full frame rows={len(X)} cols={X.shape[1]}")

    test_size = float(row.get("test_size", 0.2))
    random_seed = int(row.get("random_seed", 42))
    stratify_labels = safe_stratify(y, bool(row.get("stratified_split", True)))

    log_step(dataset_name, "building train/test frames for explainability")
    X_train, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_labels,
    )
    X_background = sample_frame(X_train, background_sample_size)
    X_explain = sample_frame(X_test, explain_sample_size)
    log_step(dataset_name, f"background frame rows={len(X_background)} cols={X_background.shape[1]}")
    log_step(dataset_name, f"explain frame rows={len(X_explain)} cols={X_explain.shape[1]}")

    log_step(dataset_name, f"loading model from {model_path.name}")
    model = joblib.load(model_path)
    output_dir = RESULTS_FIGURES_DIR / "shap" / dataset_name
    feature_manifest_path = _write_feature_manifest(output_dir, dataset_name, rename_map, feature_metadata)

    log_step(dataset_name, f"running global explainability in mode={mode}")
    global_outputs = run_global_shap(
        model=model,
        X_background=X_background,
        X_explain=X_explain,
        output_dir=output_dir,
        dataset_name=dataset_name,
        mode=mode,
        enable_plots=enable_plots,
        allow_fallback=allow_fallback,
    )
    log_step(dataset_name, f"running local explainability in mode={mode}")
    local_outputs = run_local_shap(
        model=model,
        X_reference=X_background,
        X_row=X_explain.iloc[[0]],
        output_dir=output_dir,
        dataset_name=dataset_name,
        row_label="test_row_0",
        mode=mode,
        allow_fallback=allow_fallback,
    )

    logger.info("Saved SHAP global outputs for %s: %s", dataset_name, global_outputs)
    logger.info("Saved SHAP local outputs for %s: %s", dataset_name, local_outputs)
    return _build_shap_record(row, model_path, global_outputs, local_outputs, feature_metadata, X, feature_manifest_path)


def main() -> None:
    ensure_project_dirs()
    explainability_config = load_explainability_config()
    best_models_df = load_final_models_table()
    logger.info("Loaded %s final-model row(s) for SHAP.", len(best_models_df))
    logger.info("Explainability config: %s", explainability_config)

    records: list[dict[str, Any]] = []
    for _, row in best_models_df.iterrows():
        logger.info("Running SHAP for dataset=%s using model=%s", row["dataset_name"], row["model"])
        records.append(run_dataset_shap(row, explainability_config))

    summary_df = pd.DataFrame(records)
    write_csv(summary_df, SHAP_SUMMARY_PATH)
    write_json(
        {
            "source_final_models": str(FINAL_MODELS_PATH if FINAL_MODELS_PATH.exists() else BEST_MODELS_PATH),
            "summary_table": str(SHAP_SUMMARY_PATH),
            "explainability_config": explainability_config,
            "per_dataset_outputs": records,
            "artifact_schema_version": "paper-v1",
        },
        SHAP_MANIFEST_PATH,
    )
    logger.info("Saved SHAP summary to %s", SHAP_SUMMARY_PATH)
    logger.info("Saved SHAP manifest to %s", SHAP_MANIFEST_PATH)


if __name__ == "__main__":
    main()
