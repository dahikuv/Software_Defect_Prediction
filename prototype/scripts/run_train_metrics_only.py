"""Run metrics-only baseline training experiments."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data.split import reconstruct_split_frames
from src.evaluation.compare import build_results_table, rank_models_by_dataset, summarize_results_table
from src.features.metrics_features import fit_metrics_feature_spec, transform_metrics_training_frame
from src.models.trainer import save_model, train_and_evaluate_model
from src.utils.io import read_parquet, write_csv, write_json
from src.utils.logging import get_logger
from src.utils.paths import CONFIG_PATH, MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_TABLES_DIR, SPLITS_DIR, ensure_project_dirs
from src.utils.provenance import artifact_uses_commit_text
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

PRIMARY_DATASET_NAMES = ["cm1", "jm1", "kc1", "pc1"]
JITLINE_DATASET_NAMES = ["openstack", "qt", "jitfine"]
RESULTS_TABLE_PATH = RESULTS_TABLES_DIR / "results_table.csv"
SUMMARY_TABLE_PATH = RESULTS_TABLES_DIR / "results_summary.csv"
RANKED_RESULTS_PATH = RESULTS_TABLES_DIR / "results_ranked.csv"
TRAINING_FAILURES_PATH = RESULTS_TABLES_DIR / "training_failures.csv"
EXPERIMENT_MANIFEST_PATH = RESULTS_TABLES_DIR / "training_manifest.json"
METRICS_MODELS_DIR = MODELS_DIR / "metrics"
THRESHOLD_PRECISION_FLOOR = 0.30
TRAINING_FAILURE_COLUMNS = ["dataset_name", "stage", "error", "random_seed", "model", "source_file", "split_mode"]

def load_training_config() -> dict[str, Any]:
    """Load training-related settings from the project config file."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_experiment_artifact(dataset_name: str) -> Path:
    """Return the experiment-ready parquet path for one dataset."""
    return PROCESSED_DATA_DIR / "experiments" / f"{dataset_name}_experiment.parquet"

def load_clean_artifact(dataset_name: str) -> Path:
    """Return the cleaned parquet path for one dataset."""
    return PROCESSED_DATA_DIR / f"{dataset_name}_clean.parquet"

def load_split_paths(dataset_name: str) -> dict[str, Path]:
    """Return split artifact paths for one dataset."""
    dataset_dir = SPLITS_DIR / dataset_name
    return {
        "split_manifest_path": dataset_dir / "manifest.json",
        "train_ids_path": dataset_dir / "train_ids.csv",
        "val_ids_path": dataset_dir / "val_ids.csv",
        "test_ids_path": dataset_dir / "test_ids.csv",
    }

def resolve_metrics_for_dataset(config: dict[str, Any], dataset_name: str, default_metrics: list[str]) -> list[str]:
    """Pick metric columns for a dataset from `features.metrics_by_dataset` or fall back to the global list."""
    by_dataset = config.get("features", {}).get("metrics_by_dataset", {}) or {}
    override = by_dataset.get(dataset_name)
    if override:
        return [str(metric) for metric in override]
    return list(default_metrics)

def resolve_threshold_strategy(config: dict[str, Any], dataset_name: str) -> str:
    """Resolve the validation-threshold strategy from config (with per-dataset override)."""
    training_cfg = config.get("training", {}) or {}
    overrides = training_cfg.get("threshold_strategy_by_dataset", {}) or {}
    override = overrides.get(dataset_name)
    if isinstance(override, str) and override.strip():
        return override.strip()
    default = training_cfg.get("threshold_strategy", "recall_with_precision_floor")
    return str(default).strip() or "recall_with_precision_floor"

def discover_jitline_datasets() -> list[str]:
    """Return JITLine dataset names whose cleaned parquet exists locally."""
    return [name for name in JITLINE_DATASET_NAMES if load_clean_artifact(name).exists()]

def build_run_manifest(config: dict[str, Any], jitline_datasets: list[str]) -> dict[str, Any]:
    """Build top-level metadata for the training run."""
    return {
        "random_seed": int(config.get("project", {}).get("random_seed", 42)),
        "test_size": float(config.get("split", {}).get("test_size", 0.2)),
        "val_size": float(config.get("split", {}).get("val_size", 0.1)),
        "models": list(config.get("models", {}).get("candidates", ["rf"])),
        "metrics": list(config.get("features", {}).get("metrics", [])),
        "metrics_by_dataset": dict(config.get("features", {}).get("metrics_by_dataset", {}) or {}),
        "datasets": list(PRIMARY_DATASET_NAMES) + list(jitline_datasets),
        "primary_datasets": list(PRIMARY_DATASET_NAMES),
        "jitline_datasets": list(jitline_datasets),
        "feature_mode": "metrics_only",
        "feature_family": "metrics_only",
        "split_source": "phase_6_manifests_or_jitline_native",
        "threshold_selection_strategy": "max_recall_with_precision_floor",
        "threshold_precision_floor": THRESHOLD_PRECISION_FLOOR,
        "selection_data_source": "validation",
        "test_metrics_report_only": True,
        "artifact_schema_version": "paper-v1",
    }

def _should_stratify(labels: pd.Series, use_stratify: bool = True) -> bool:
    counts = labels.value_counts(dropna=False)
    return bool(use_stratify and labels.nunique(dropna=True) > 1 and not counts.empty and counts.min() >= 2)

def _fresh_three_way_split(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a stratified train/val/test split when no saved manifest is used."""
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError("Fresh split requires 0 < test_size, val_size and test_size + val_size < 1")
    holdout_size = test_size + val_size
    stratify_outer = df["label"] if _should_stratify(df["label"]) else None
    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_size,
        random_state=random_seed,
        stratify=stratify_outer,
    )
    stratify_inner = holdout_df["label"] if _should_stratify(holdout_df["label"]) else None
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=test_size / holdout_size,
        random_state=random_seed,
        stratify=stratify_inner,
    )
    overlap = (
        set(train_df.index)
        .intersection(val_df.index)
        .union(set(train_df.index).intersection(test_df.index))
        .union(set(val_df.index).intersection(test_df.index))
    )
    if overlap:
        raise ValueError(f"Fresh split produced overlapping indices: count={len(overlap)}")
    return train_df.copy(), val_df.copy(), test_df.copy()


def _split_native_jitline_frames(
    df: pd.DataFrame,
    val_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use the native train/(val/)test marker, carving validation off the train pool when missing."""
    if "jitline_split" not in df.columns:
        raise ValueError("Native-split dataset is missing the jitline_split column required for the native split.")

    normalized = df["jitline_split"].astype(str).str.strip().str.lower()
    train_pool = df.loc[normalized == "train"].copy()
    test_df = df.loc[normalized == "test"].copy()
    native_val_df = df.loc[normalized.isin({"val", "valid", "validation"})].copy()
    if train_pool.empty or test_df.empty:
        raise ValueError("Native split needs both train and test rows.")

    if not native_val_df.empty:
        return train_pool.copy(), native_val_df.copy(), test_df.copy()

    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1 for native split when no native val partition exists.")
    stratify_labels = train_pool["label"] if _should_stratify(train_pool["label"]) else None
    train_df, val_df = train_test_split(
        train_pool,
        test_size=val_size,
        random_state=random_seed,
        stratify=stratify_labels,
    )
    return train_df.copy(), val_df.copy(), test_df.copy()

def _normalize_result_record(result: dict[str, Any], dataset_name: str, model_name: str, artifact_stage: str, source_file: str, split_manifest_path: Path, split_manifest_hash: str, split_mode: str, random_seed: int, test_size: float, val_size: float, use_saved_splits: bool, model_path: Path, feature_metadata: dict[str, Any], metrics: list[str], train_df: Any, val_df: Any, test_df: Any) -> dict[str, Any]:
    feature_family = str(feature_metadata.get("feature_family") or feature_metadata.get("feature_set") or "metrics_only")
    normalized = {
        **result,
        "dataset_name": dataset_name,
        "model": model_name,
        "feature_family": feature_family,
        "feature_set": feature_metadata.get("feature_set", feature_family),
        "text_feature_column": feature_metadata.get("text_feature_column", ""),
        "uses_commit_text": artifact_uses_commit_text(feature_metadata),
        "artifact_stage": artifact_stage,
        "artifact_schema_version": "paper-v1",
        "artifact_created_at": result.get("artifact_created_at", ""),
        "artifact_group_key": f"{dataset_name}::{model_name}",
        "artifact_id": f"{dataset_name}::{model_name}::{artifact_stage}",
        "source_results_table": str(RESULTS_TABLE_PATH),
        "source_file": source_file,
        "split_manifest_path": str(split_manifest_path) if split_manifest_path else "",
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

def validate_required_artifacts(use_saved_splits: bool) -> None:
    """Fail early when experiment datasets or saved splits are missing for the PROMISE family."""
    missing_experiments = [
        str(load_experiment_artifact(dataset_name))
        for dataset_name in PRIMARY_DATASET_NAMES
        if not load_experiment_artifact(dataset_name).exists()
    ]
    missing_split_files: list[str] = []
    if use_saved_splits:
        for dataset_name in PRIMARY_DATASET_NAMES:
            missing_split_files.extend(str(path) for path in load_split_paths(dataset_name).values() if not path.exists())

    if missing_experiments or missing_split_files:
        details = []
        if missing_experiments:
            details.append(f"missing experiment artifact(s): {missing_experiments}")
        if missing_split_files:
            details.append(f"missing split artifact(s): {missing_split_files}")
        raise FileNotFoundError(
            "Metrics training requires experiment datasets and saved splits for PROMISE baselines. "
            "Run `python prototype\\scripts\\run_experiment_datasets.py` first. "
            + "; ".join(details)
        )

def _train_one_dataset(
    *,
    dataset_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: list[str],
    model_candidates: list[str],
    random_seed: int,
    test_size: float,
    val_size: float,
    use_saved_splits: bool,
    split_mode: str,
    split_manifest_path: Path | None,
    split_manifest_hash: str,
    source_file: str,
    threshold_strategy: str = "recall_with_precision_floor",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit metrics-only models for one dataset and return result/failure records."""
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if train_df.empty or test_df.empty:
        failures.append(
            _make_empty_failure(
                dataset_name,
                "split_validation",
                "Train or test split is empty.",
                random_seed,
                source_file=source_file,
                split_mode=split_mode,
            )
        )
        return results, failures

    try:
        feature_spec = fit_metrics_feature_spec(train_df, metrics)
        X_train, y_train, feature_metadata = transform_metrics_training_frame(train_df, feature_spec)
        if val_df.empty:
            X_val = X_train.iloc[0:0].copy()
            y_val = y_train.iloc[0:0].copy()
        else:
            X_val, y_val, _ = transform_metrics_training_frame(val_df, feature_spec)
        X_test, y_test, _ = transform_metrics_training_frame(test_df, feature_spec)
    except Exception as exc:
        failures.append(
            _make_empty_failure(
                dataset_name,
                "feature_building",
                str(exc),
                random_seed,
                source_file=source_file,
                split_mode=split_mode,
            )
        )
        return results, failures

    if X_train.empty or X_train.shape[1] == 0:
        failures.append(
            _make_empty_failure(
                dataset_name,
                "feature_building",
                "No usable metrics features found in train split.",
                random_seed,
                source_file=source_file,
                split_mode=split_mode,
            )
        )
        return results, failures
    if y_train.nunique() < 2:
        failures.append(
            _make_empty_failure(
                dataset_name,
                "dataset_validation",
                "Train split needs at least two classes.",
                random_seed,
                source_file=source_file,
                split_mode=split_mode,
            )
        )
        return results, failures
    if X_test.empty or X_test.shape[1] == 0:
        failures.append(
            _make_empty_failure(
                dataset_name,
                "feature_building",
                "No usable metrics features found in test split.",
                random_seed,
                source_file=source_file,
                split_mode=split_mode,
            )
        )
        return results, failures
    if y_test.nunique() < 2:
        failures.append(
            _make_empty_failure(
                dataset_name,
                "dataset_validation",
                "Test split needs at least two classes.",
                random_seed,
                source_file=source_file,
                split_mode=split_mode,
            )
        )
        return results, failures

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
                X_val=X_val if not X_val.empty else None,
                y_val=y_val if not y_val.empty else None,
                threshold_precision_floor=THRESHOLD_PRECISION_FLOOR,
                threshold_strategy=threshold_strategy,
                feature_preprocessor=feature_spec,
            )
            model_path = METRICS_MODELS_DIR / f"{model_name}_{dataset_name}.joblib"
            save_model(model, model_path)
            normalized_result = _normalize_result_record(
                result=result,
                dataset_name=dataset_name,
                model_name=model_name,
                artifact_stage="training",
                source_file=source_file,
                split_manifest_path=split_manifest_path if split_manifest_path else Path(""),
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
            results.append(normalized_result)
        except Exception as exc:
            failures.append(
                _make_empty_failure(
                    dataset_name,
                    "model_training",
                    str(exc),
                    random_seed,
                    model=model_name,
                    source_file=source_file,
                    split_mode=split_mode,
                )
            )
    return results, failures

def main() -> None:
    """Execute the full metrics-only training flow for PROMISE + JITLine datasets."""
    ensure_project_dirs()
    config = load_training_config()
    random_seed = int(config.get("project", {}).get("random_seed", 42))
    default_metrics = list(config.get("features", {}).get("metrics", []))
    model_candidates = list(config.get("models", {}).get("candidates", ["rf"]))
    test_size = float(config.get("split", {}).get("test_size", 0.2))
    val_size = float(config.get("split", {}).get("val_size", 0.1))
    use_saved_splits = bool(config.get("split", {}).get("use_saved_splits", True))
    validate_required_artifacts(use_saved_splits)

    set_global_seed(random_seed)
    jitline_datasets = discover_jitline_datasets()
    logger.info(
        "Training metrics-only models with seed=%s, test_size=%s, val_size=%s, models=%s, jitline=%s",
        random_seed,
        test_size,
        val_size,
        ", ".join(model_candidates),
        ", ".join(jitline_datasets) or "none",
    )

    run_manifest = build_run_manifest(config, jitline_datasets)
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
            try:
                train_df, val_df, test_df = _fresh_three_way_split(
                    df,
                    test_size=test_size,
                    val_size=val_size,
                    random_seed=random_seed,
                )
            except Exception as exc:
                all_failures.append(
                    _make_empty_failure(
                        dataset_name,
                        "split_loading",
                        f"fresh split failed: {exc}",
                        random_seed,
                        source_file=str(artifact_path),
                    )
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

        dataset_metrics = resolve_metrics_for_dataset(config, dataset_name, default_metrics)
        results, failures = _train_one_dataset(
            dataset_name=dataset_name,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            metrics=dataset_metrics,
            model_candidates=model_candidates,
            random_seed=random_seed,
            test_size=test_size,
            val_size=val_size,
            use_saved_splits=use_saved_splits,
            split_mode=split_mode,
            split_manifest_path=split_manifest_path,
            split_manifest_hash=split_manifest_hash,
            source_file=str(artifact_path),
            threshold_strategy=resolve_threshold_strategy(config, dataset_name),
        )
        all_results.extend(results)
        all_failures.extend(failures)

    for dataset_name in jitline_datasets:
        clean_path = load_clean_artifact(dataset_name)
        try:
            df = read_parquet(clean_path)
        except Exception as exc:
            all_failures.append(
                _make_empty_failure(
                    dataset_name,
                    "dataset_loading",
                    f"Failed to read JITLine cleaned parquet: {exc}",
                    random_seed,
                    source_file=str(clean_path),
                    split_mode="jitline_native_split",
                )
            )
            continue

        if df.empty:
            all_failures.append(
                _make_empty_failure(
                    dataset_name,
                    "dataset_validation",
                    "JITLine cleaned parquet is empty.",
                    random_seed,
                    source_file=str(clean_path),
                    split_mode="jitline_native_split",
                )
            )
            continue

        try:
            train_df, val_df, test_df = _split_native_jitline_frames(df, val_size=val_size, random_seed=random_seed)
        except Exception as exc:
            all_failures.append(
                _make_empty_failure(
                    dataset_name,
                    "split_loading",
                    str(exc),
                    random_seed,
                    source_file=str(clean_path),
                    split_mode="jitline_native_split",
                )
            )
            continue

        dataset_metrics = resolve_metrics_for_dataset(config, dataset_name, default_metrics)
        results, failures = _train_one_dataset(
            dataset_name=dataset_name,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            metrics=dataset_metrics,
            model_candidates=model_candidates,
            random_seed=random_seed,
            test_size=test_size,
            val_size=val_size,
            use_saved_splits=False,
            split_mode="jitline_native_split",
            split_manifest_path=None,
            split_manifest_hash="",
            source_file=str(clean_path),
            threshold_strategy=resolve_threshold_strategy(config, dataset_name),
        )
        all_results.extend(results)
        all_failures.extend(failures)

    results_df = build_results_table(all_results)
    failures_df = build_results_table(all_failures)
    if failures_df.empty and len(failures_df.columns) == 0:
        failures_df = pd.DataFrame(columns=TRAINING_FAILURE_COLUMNS)
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
