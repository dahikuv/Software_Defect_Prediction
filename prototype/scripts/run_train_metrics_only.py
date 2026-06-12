"""Run metrics-only baseline training experiments."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.data.split import reconstruct_split_frames
from src.evaluation.compare import build_results_table, rank_models_by_dataset, summarize_results_table
from src.features.metrics_features import build_metrics_training_frame
from src.models.trainer import save_model, train_and_evaluate_model
from src.utils.io import read_parquet, write_csv, write_json
from src.utils.logging import get_logger
from src.utils.paths import CONFIG_PATH, MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_TABLES_DIR, SPLITS_DIR, ensure_project_dirs
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

PRIMARY_DATASET_NAMES = ["cm1", "jm1", "kc1", "pc1"]
RESULTS_TABLE_PATH = RESULTS_TABLES_DIR / "results_table.csv"
SUMMARY_TABLE_PATH = RESULTS_TABLES_DIR / "results_summary.csv"
RANKED_RESULTS_PATH = RESULTS_TABLES_DIR / "results_ranked.csv"
TRAINING_FAILURES_PATH = RESULTS_TABLES_DIR / "training_failures.csv"
EXPERIMENT_MANIFEST_PATH = RESULTS_TABLES_DIR / "training_manifest.json"
METRICS_MODELS_DIR = MODELS_DIR / "metrics"


def load_training_config() -> dict[str, Any]:
    """Load training-related settings from the project config file."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_experiment_artifact(dataset_name: str) -> Path:
    """Return the experiment-ready parquet path for one dataset."""
    return PROCESSED_DATA_DIR / "experiments" / f"{dataset_name}_experiment.parquet"


def load_split_paths(dataset_name: str) -> dict[str, Path]:
    """Return split artifact paths for one dataset."""
    dataset_dir = SPLITS_DIR / dataset_name
    return {
        "split_manifest_path": dataset_dir / "manifest.json",
        "train_ids_path": dataset_dir / "train_ids.csv",
        "val_ids_path": dataset_dir / "val_ids.csv",
        "test_ids_path": dataset_dir / "test_ids.csv",
    }


def build_run_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """Build top-level metadata for the training run."""
    return {
        "random_seed": int(config.get("project", {}).get("random_seed", 42)),
        "test_size": float(config.get("split", {}).get("test_size", 0.2)),
        "val_size": float(config.get("split", {}).get("val_size", 0.1)),
        "models": list(config.get("models", {}).get("candidates", ["rf"])),
        "metrics": list(config.get("features", {}).get("metrics", [])),
        "datasets": PRIMARY_DATASET_NAMES,
        "feature_mode": "metrics_only",
        "feature_family": "metrics_only",
        "split_source": "phase_6_manifests",
        "artifact_schema_version": "paper-v1",
    }



def _normalize_result_record(result: dict[str, Any], dataset_name: str, model_name: str, artifact_stage: str, source_file: str, split_manifest_path: Path, split_manifest_hash: str, split_mode: str, random_seed: int, test_size: float, val_size: float, use_saved_splits: bool, model_path: Path, feature_metadata: dict[str, Any], metrics: list[str], train_df: Any, val_df: Any, test_df: Any) -> dict[str, Any]:
    feature_family = str(feature_metadata.get("feature_family") or feature_metadata.get("feature_set") or "metrics_only")
    normalized = {
        **result,
        "dataset_name": dataset_name,
        "model": model_name,
        "feature_family": feature_family,
        "feature_set": feature_metadata.get("feature_set", feature_family),
        "text_feature_column": feature_metadata.get("text_feature_column", ""),
        "uses_commit_text": bool(feature_metadata.get("uses_commit_text", feature_family in {"metrics_plus_commit_text", "metrics_plus_text", "hybrid"})),
        "artifact_stage": artifact_stage,
        "artifact_schema_version": "paper-v1",
        "artifact_created_at": result.get("artifact_created_at", ""),
        "artifact_group_key": f"{dataset_name}::{model_name}",
        "artifact_id": f"{dataset_name}::{model_name}::{artifact_stage}",
        "source_results_table": str(RESULTS_TABLE_PATH),
        "source_file": source_file,
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_hash": split_manifest_hash,
        "split_mode": split_mode,
        "random_seed": random_seed,
        "test_size": test_size,
        "val_size": val_size,
        "use_saved_splits": use_saved_splits,
        "model_path": str(model_path),
        "feature_mode": "metrics_only",
        "configured_models": ",".join(model_name for model_name in [model_name]),
        "configured_metrics": ",".join(metrics),
        "num_train_rows": int(len(train_df)),
        "num_val_rows": int(len(val_df)),
        "num_test_rows": int(len(test_df)),
    }
    return normalized


def _make_empty_failure(dataset_name: str, stage: str, error: str, random_seed: int, **extra: Any) -> dict[str, Any]:
    record = {
        "dataset_name": dataset_name,
        "stage": stage,
        "error": error,
        "random_seed": random_seed,
    }
    record.update(extra)
    return record


def main() -> None:
    """Execute the full metrics-only training flow."""
    ensure_project_dirs()
    config = load_training_config()
    random_seed = int(config.get("project", {}).get("random_seed", 42))
    metrics = list(config.get("features", {}).get("metrics", []))
    model_candidates = list(config.get("models", {}).get("candidates", ["rf"]))
    test_size = float(config.get("split", {}).get("test_size", 0.2))
    val_size = float(config.get("split", {}).get("val_size", 0.1))
    use_saved_splits = bool(config.get("split", {}).get("use_saved_splits", True))

    set_global_seed(random_seed)
    logger.info(
        "Training metrics-only models with seed=%s, test_size=%s, val_size=%s, models=%s",
        random_seed,
        test_size,
        val_size,
        ", ".join(model_candidates),
    )

    run_manifest = build_run_manifest(config)
    all_results: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []

    for dataset_name in PRIMARY_DATASET_NAMES:
        artifact_path = load_experiment_artifact(dataset_name)
        split_paths = load_split_paths(dataset_name)
        split_manifest_path = split_paths["split_manifest_path"]
        if not artifact_path.exists():
            all_failures.append(
                _make_empty_failure(
                    dataset_name,
                    "dataset_loading",
                    f"Missing experiment artifact: {artifact_path}",
                    random_seed,
                    source_file=str(artifact_path),
                )
            )
            continue

        df = read_parquet(artifact_path)
        if df.empty:
            all_failures.append(
                _make_empty_failure(
                    dataset_name,
                    "dataset_validation",
                    "Experiment artifact is empty.",
                    random_seed,
                    source_file=str(artifact_path),
                )
            )
            continue

        if use_saved_splits:
            missing_split_files = [str(path) for path in split_paths.values() if not path.exists()]
            if missing_split_files:
                all_failures.append(
                    _make_empty_failure(
                        dataset_name,
                        "split_loading",
                        f"Missing split artifact(s): {missing_split_files}",
                        random_seed,
                        source_file=str(artifact_path),
                    )
                )
                continue
            try:
                train_df, val_df, test_df = reconstruct_split_frames(
                    df,
                    split_paths["train_ids_path"],
                    split_paths["val_ids_path"],
                    split_paths["test_ids_path"],
                )
            except Exception as exc:
                all_failures.append(
                    _make_empty_failure(
                        dataset_name,
                        "split_loading",
                        str(exc),
                        random_seed,
                        source_file=str(artifact_path),
                    )
                )
                continue
        else:
            train_df = df.copy()
            val_df = df.iloc[0:0].copy()
            test_df = df.copy()

        if train_df.empty or test_df.empty:
            all_failures.append(
                _make_empty_failure(
                    dataset_name,
                    "split_validation",
                    "Train or test split is empty.",
                    random_seed,
                    source_file=str(artifact_path),
                    split_mode="saved_split" if use_saved_splits else "fresh_split",
                )
            )
            continue

        try:
            X_train, y_train, feature_metadata = build_metrics_training_frame(train_df, metrics)
            X_test, y_test, _ = build_metrics_training_frame(test_df, metrics)
        except Exception as exc:
            all_failures.append(
                _make_empty_failure(
                    dataset_name,
                    "feature_building",
                    str(exc),
                    random_seed,
                    source_file=str(artifact_path),
                )
            )
            continue

        if X_train.empty or X_train.shape[1] == 0:
            all_failures.append(
                _make_empty_failure(dataset_name, "feature_building", "No usable metrics features found in train split.", random_seed, source_file=str(artifact_path))
            )
            continue
        if y_train.nunique() < 2:
            all_failures.append(
                _make_empty_failure(dataset_name, "dataset_validation", "Train split needs at least two classes.", random_seed, source_file=str(artifact_path))
            )
            continue
        if X_test.empty or X_test.shape[1] == 0:
            all_failures.append(
                _make_empty_failure(dataset_name, "feature_building", "No usable metrics features found in test split.", random_seed, source_file=str(artifact_path))
            )
            continue
        if y_test.nunique() < 2:
            all_failures.append(
                _make_empty_failure(dataset_name, "dataset_validation", "Test split needs at least two classes.", random_seed, source_file=str(artifact_path))
            )
            continue

        split_mode = "saved_split" if use_saved_splits else "fresh_split"
        split_manifest_hash = ""
        if split_manifest_path.exists():
            try:
                split_manifest = yaml.safe_load(split_manifest_path.read_text(encoding="utf-8")) or {}
                split_manifest_hash = str(split_manifest.get("manifest_hash", ""))
            except Exception:
                split_manifest_hash = ""

        for model_name in model_candidates:
            try:
                model, result = train_and_evaluate_model(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    random_state=random_seed,
                    feature_metadata=feature_metadata,
                )
                model_path = METRICS_MODELS_DIR / f"{model_name}_{dataset_name}.joblib"
                save_model(model, model_path)
                normalized_result = _normalize_result_record(
                    result=result,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    artifact_stage="training",
                    source_file=str(artifact_path),
                    split_manifest_path=split_manifest_path,
                    split_manifest_hash=split_manifest_hash,
                    split_mode=split_mode,
                    random_seed=random_seed,
                    test_size=test_size,
                    val_size=val_size,
                    use_saved_splits=use_saved_splits,
                    model_path=model_path,
                    feature_metadata=feature_metadata,
                    metrics=metrics,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                )
                normalized_result["configured_models"] = ",".join(model_candidates)
                normalized_result["configured_metrics"] = ",".join(metrics)
                normalized_result["feature_mode"] = "metrics_only"
                all_results.append(normalized_result)
            except Exception as exc:
                all_failures.append(
                    _make_empty_failure(
                        dataset_name,
                        "model_training",
                        str(exc),
                        random_seed,
                        model=model_name,
                        source_file=str(artifact_path),
                    )
                )

    results_df = build_results_table(all_results)
    failures_df = build_results_table(all_failures)
    write_csv(results_df, RESULTS_TABLE_PATH)
    write_csv(summarize_results_table(results_df), SUMMARY_TABLE_PATH)
    write_csv(rank_models_by_dataset(results_df), RANKED_RESULTS_PATH)
    write_csv(failures_df, TRAINING_FAILURES_PATH)
    write_json(run_manifest, EXPERIMENT_MANIFEST_PATH)

    logger.info("Saved results table to %s", RESULTS_TABLE_PATH)
    logger.info("Saved summary table to %s", SUMMARY_TABLE_PATH)
    logger.info("Saved ranked table to %s", RANKED_RESULTS_PATH)
    logger.info("Saved training failures table to %s", TRAINING_FAILURES_PATH)
    logger.info("Saved training manifest to %s", EXPERIMENT_MANIFEST_PATH)


if __name__ == "__main__":
    main()
