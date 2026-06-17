"""Run the SHAP scaffold."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import yaml

from src.data.split import reconstruct_split_frames
from src.explainability.shap_global import run_global_shap
from src.explainability.shap_local import run_local_shap
from src.features.metrics_features import build_metrics_training_frame
from src.models.bundle import ModelBundle
from src.utils.io import read_csv, read_parquet, write_csv, write_json
from src.utils.logging import get_logger
from src.utils.paths import CONFIG_PATH, PROCESSED_DATA_DIR, RESULTS_FIGURES_DIR, RESULTS_TABLES_DIR, SPLITS_DIR, ensure_project_dirs
from src.utils.provenance import artifact_uses_commit_text

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
    for key in ("feature_family", "feature_set"):
        value = row.get(key)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value)
    return "metrics_only"


JITLINE_DATASETS = {"openstack", "qt", "jitfine"}

def load_saved_split_frames_for_dataset(row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test split frames for one final-model row."""
    dataset_name = str(row["dataset_name"])
    dataset_path = PROCESSED_DATA_DIR / f"{dataset_name}_clean.parquet"
    df = read_parquet(dataset_path)
    split_mode = str(row.get("split_mode", "saved_split") or "saved_split")
    if dataset_name in JITLINE_DATASETS or split_mode == "jitline_native_split":
        if "jitline_split" not in df.columns:
            raise ValueError(f"Dataset {dataset_name} is missing the jitline_split column required for the native split.")
        normalized = df["jitline_split"].astype(str).str.strip().str.lower()
        train_df = df.loc[normalized == "train"].copy()
        test_df = df.loc[normalized == "test"].copy()
        if train_df.empty or test_df.empty:
            raise ValueError(f"JITLine native split for {dataset_name} produced an empty train or test frame.")
        return train_df, test_df
    split_dir = SPLITS_DIR / dataset_name
    train_df, _, test_df = reconstruct_split_frames(
        df,
        split_dir / "train_ids.csv",
        split_dir / "val_ids.csv",
        split_dir / "test_ids.csv",
    )
    return train_df, test_df


def build_feature_frame_for_split(df: pd.DataFrame, row: pd.Series) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Build a feature frame matching the selected model family."""
    available_metrics = [metric for metric in DEFAULT_METRICS if metric in df.columns]
    feature_family = _normalize_feature_family(row)

    if feature_family in HYBRID_FEATURE_FAMILIES or artifact_uses_commit_text(row.to_dict()):
        raise ValueError(
            "Legacy commit-text artifact cannot be reconstructed for SHAP without a train-fitted ModelBundle."
        )

    X, y, metadata = build_metrics_training_frame(df, available_metrics)
    metadata["resolved_feature_family"] = "metrics_only"
    return X, y, metadata


def _split_feature_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    metric_columns = [column for column in X.columns if not str(column).startswith("commit_")]
    commit_columns = [column for column in X.columns if str(column).startswith("commit_")]
    return metric_columns, commit_columns


def _sanitize_feature_names(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Convert feature names to SHAP-safe names while preserving a mapping."""
    rename_map: dict[str, str] = {}
    used_names: dict[str, int] = {}
    for column in X.columns:
        base_name = (
            str(column)
            .replace("(", "_")
            .replace(")", "_")
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
            .strip("_")
            or "feature"
        )
        count = used_names.get(base_name, 0)
        used_names[base_name] = count + 1
        rename_map[column] = base_name if count == 0 else f"{base_name}_{count + 1}"
    return X.rename(columns=rename_map), rename_map

def _apply_feature_name_map(X: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    mapped = X.rename(columns=rename_map)
    ordered_columns = [rename_map[column] for column in X.columns if column in rename_map]
    return mapped[ordered_columns].copy()

@contextmanager
def _temporary_feature_names(model: Any, feature_names: list[str]):
    original = getattr(model, "feature_names_in_", None)
    should_patch = original is not None and len(original) == len(feature_names)
    patched = False
    if should_patch:
        try:
            model.feature_names_in_ = np.asarray(feature_names, dtype=object)
            patched = True
        except Exception:
            patched = False
    try:
        yield
    finally:
        if patched:
            try:
                model.feature_names_in_ = original
            except Exception:
                pass


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
    uses_commit_text = artifact_uses_commit_text({**row.to_dict(), **feature_metadata})

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
        "decision_threshold": row.get("decision_threshold", ""),
        "split_mode": row.get("split_mode", "saved_split"),
        "split_manifest_path": row.get("split_manifest_path", ""),
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
    effective_mode = mode
    background_sample_size = int(explainability_config.get("background_sample_size", 100))
    explain_sample_size = int(explainability_config.get("explain_sample_size", 50))
    enable_plots = bool(explainability_config.get("enable_plots", False))
    allow_fallback = bool(explainability_config.get("allow_fallback", True))

    log_step(dataset_name, f"loading saved train/test split frames from {dataset_name}_clean.parquet")
    train_df, test_df = load_saved_split_frames_for_dataset(row)

    log_step(dataset_name, f"loading model from {model_path.name}")
    loaded_model = joblib.load(model_path)
    if isinstance(loaded_model, ModelBundle):
        X_train = loaded_model.transform_features(train_df)
        X_test = loaded_model.transform_features(test_df)
        feature_metadata = {
            "resolved_feature_family": loaded_model.feature_family,
            "num_features": int(X_train.shape[1]),
            "bundle_metadata": loaded_model.metadata,
        }
        model = loaded_model.estimator
    else:
        X_train, _, train_metadata = build_feature_frame_for_split(train_df, row)
        X_test, _, test_metadata = build_feature_frame_for_split(test_df, row)
        feature_metadata = {**train_metadata, "test_feature_metadata": test_metadata}
        model = loaded_model

    X_train, rename_map = _sanitize_feature_names(X_train)
    X_test = _apply_feature_name_map(X_test, rename_map)
    log_step(dataset_name, f"saved split rows train={len(X_train)} test={len(X_test)} cols={X_train.shape[1]}")

    X_background = sample_frame(X_train, background_sample_size)
    X_explain = sample_frame(X_test, explain_sample_size)
    log_step(dataset_name, f"background frame rows={len(X_background)} cols={X_background.shape[1]}")
    log_step(dataset_name, f"explain frame rows={len(X_explain)} cols={X_explain.shape[1]}")

    output_dir = RESULTS_FIGURES_DIR / "shap" / dataset_name
    feature_manifest_path = _write_feature_manifest(output_dir, dataset_name, rename_map, feature_metadata)

    with _temporary_feature_names(model, list(X_train.columns)):
        log_step(dataset_name, f"running global explainability in mode={effective_mode}")
        global_outputs = run_global_shap(
            model=model,
            X_background=X_background,
            X_explain=X_explain,
            output_dir=output_dir,
            dataset_name=dataset_name,
            mode=effective_mode,
            enable_plots=enable_plots,
            allow_fallback=allow_fallback,
        )
        log_step(dataset_name, f"running local explainability in mode={effective_mode}")
        local_outputs = run_local_shap(
            model=model,
            X_reference=X_background,
            X_row=X_explain.iloc[[0]],
            output_dir=output_dir,
            dataset_name=dataset_name,
            row_label="test_row_0",
            mode=effective_mode,
            allow_fallback=allow_fallback,
        )

    logger.info("Saved SHAP global outputs for %s: %s", dataset_name, global_outputs)
    logger.info("Saved SHAP local outputs for %s: %s", dataset_name, local_outputs)
    record = _build_shap_record(row, model_path, global_outputs, local_outputs, feature_metadata, X_train, feature_manifest_path)
    record["requested_mode"] = mode
    record["effective_mode"] = global_outputs.get("mode_used", effective_mode)
    record["local_effective_mode"] = local_outputs.get("mode_used", effective_mode)
    return record


def _build_shap_failure_record(row: pd.Series, error: Exception) -> dict[str, Any]:
    dataset_name = str(row.get("dataset_name", ""))
    model_name = str(row.get("model", ""))
    return {
        "dataset_name": dataset_name,
        "model": model_name,
        "model_path": str(row.get("model_path", "")),
        "feature_family": _normalize_feature_family(row),
        "feature_set": str(row.get("feature_set", row.get("feature_family", ""))),
        "text_feature_column": str(row.get("text_feature_column", "")),
        "uses_commit_text": artifact_uses_commit_text(row.to_dict()),
        "artifact_schema_version": str(row.get("artifact_schema_version", "paper-v1")),
        "artifact_stage": "shap",
        "artifact_id": str(row.get("artifact_id", f"{dataset_name}::{model_name}::shap")),
        "status": "failed",
        "error": str(error),
    }


def main() -> None:
    ensure_project_dirs()
    explainability_config = load_explainability_config()
    best_models_df = load_final_models_table()
    logger.info("Loaded %s final-model row(s) for SHAP.", len(best_models_df))
    logger.info("Explainability config: %s", explainability_config)

    records: list[dict[str, Any]] = []
    for _, row in best_models_df.iterrows():
        logger.info("Running SHAP for dataset=%s using model=%s", row["dataset_name"], row["model"])
        try:
            records.append(run_dataset_shap(row, explainability_config))
        except Exception as exc:
            logger.exception("SHAP failed for dataset=%s model=%s: %s", row.get("dataset_name"), row.get("model"), exc)
            records.append(_build_shap_failure_record(row, exc))

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
