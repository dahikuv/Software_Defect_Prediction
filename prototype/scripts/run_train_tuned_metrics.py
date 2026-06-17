"""Run tuned metrics-only experiments on fixed train/validation/test splits."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.data.split import reconstruct_split_frames
from src.evaluation.compare import build_results_table, rank_models_by_dataset, summarize_results_table
from src.features.metrics_features import fit_metrics_feature_spec, transform_metrics_training_frame
from src.models.bundle import ModelBundle
from src.models.registry import get_model
from src.models.trainer import configure_model_for_imbalance, evaluate_model, resolve_decision_threshold, save_model
from src.utils.io import read_parquet, write_csv, write_json
from src.utils.coercion import coerce_bool
from src.utils.logging import get_logger
from src.utils.paths import CONFIG_PATH, MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_TABLES_DIR, SPLITS_DIR, ensure_project_dirs
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

PRIMARY_DATASET_NAMES = {"cm1", "jm1", "kc1", "pc1"}
JITLINE_DATASETS = {"openstack", "qt", "jitfine"}
ELIGIBLE_TUNED_DATASETS = PRIMARY_DATASET_NAMES | JITLINE_DATASETS
TUNED_MODELS_DIR = MODELS_DIR / "metrics_tuned"
TUNING_RESULTS_PATH = RESULTS_TABLES_DIR / "metrics_tuned_results.csv"
TUNING_CANDIDATES_PATH = RESULTS_TABLES_DIR / "metrics_tuning_candidates.csv"
TUNING_BEST_PATH = RESULTS_TABLES_DIR / "metrics_tuned_best.csv"
TUNING_SUMMARY_PATH = RESULTS_TABLES_DIR / "metrics_tuned_summary.csv"
TUNING_RANKING_PATH = RESULTS_TABLES_DIR / "metrics_tuned_ranking.csv"
TUNING_FAILURES_PATH = RESULTS_TABLES_DIR / "metrics_tuned_failures.csv"
TUNING_CONFIG_PATH = RESULTS_TABLES_DIR / "metrics_tuned_config.json"
THRESHOLD_PRECISION_FLOOR = 0.30
DEFAULT_CV_FOLDS = 5
TUNING_FAILURE_COLUMNS = ["dataset_name", "model", "candidate_index", "stage", "error", "source_file"]


PARAMETER_GRID: dict[str, list[dict[str, Any]]] = {
    "rf": [
        {"n_estimators": 250, "max_depth": 6, "min_samples_split": 8, "min_samples_leaf": 4, "max_features": "sqrt"},
        {"n_estimators": 300, "max_depth": 8, "min_samples_split": 10, "min_samples_leaf": 5, "max_features": "sqrt"},
        {"n_estimators": 350, "max_depth": 10, "min_samples_split": 12, "min_samples_leaf": 6, "max_features": "log2"},
    ],
    "xgb": [
        {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.05, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0},
        {"n_estimators": 250, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 3.0},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "min_child_weight": 5, "subsample": 0.75, "colsample_bytree": 0.75, "reg_lambda": 4.0},
    ],
    "lgbm": [
        {"n_estimators": 200, "num_leaves": 7, "max_depth": 3, "learning_rate": 0.05, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0},
        {"n_estimators": 250, "num_leaves": 15, "max_depth": 4, "learning_rate": 0.05, "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 3.0},
        {"n_estimators": 300, "num_leaves": 15, "max_depth": 5, "learning_rate": 0.03, "min_child_samples": 30, "subsample": 0.75, "colsample_bytree": 0.75, "reg_lambda": 4.0},
    ],
}

# Smaller grid for tiny datasets (PROMISE cm1/pc1 with ~80-150 train rows). Trees stay shallow
# and we lower the count of estimators so cross-validation does not overfit, while still giving
# the tuner three diverse candidates per model.
PARAMETER_GRID_SMALL: dict[str, list[dict[str, Any]]] = {
    "rf": [
        {"n_estimators": 150, "max_depth": 4, "min_samples_split": 6, "min_samples_leaf": 3, "max_features": "sqrt"},
        {"n_estimators": 200, "max_depth": 5, "min_samples_split": 8, "min_samples_leaf": 4, "max_features": "sqrt"},
        {"n_estimators": 250, "max_depth": 6, "min_samples_split": 10, "min_samples_leaf": 5, "max_features": "log2"},
    ],
    "xgb": [
        {"n_estimators": 120, "max_depth": 2, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 4.0},
        {"n_estimators": 150, "max_depth": 2, "learning_rate": 0.04, "min_child_weight": 6, "subsample": 0.75, "colsample_bytree": 0.75, "reg_lambda": 5.0},
        {"n_estimators": 180, "max_depth": 3, "learning_rate": 0.03, "min_child_weight": 8, "subsample": 0.7, "colsample_bytree": 0.7, "reg_lambda": 6.0},
    ],
    "lgbm": [
        {"n_estimators": 120, "num_leaves": 7, "max_depth": 3, "learning_rate": 0.05, "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 4.0},
        {"n_estimators": 150, "num_leaves": 7, "max_depth": 3, "learning_rate": 0.04, "min_child_samples": 25, "subsample": 0.75, "colsample_bytree": 0.75, "reg_lambda": 5.0},
        {"n_estimators": 180, "num_leaves": 11, "max_depth": 4, "learning_rate": 0.03, "min_child_samples": 30, "subsample": 0.7, "colsample_bytree": 0.7, "reg_lambda": 6.0},
    ],
}

DEFAULT_SMALL_GRID_THRESHOLD = 1500


def select_parameter_grid(
    config: dict[str, Any],
    dataset_name: str,
    n_train: int,
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Pick the param grid for a dataset, optionally driven by config knobs.

    Resolution order:
    1. ``training.parameter_grid_by_dataset[dataset]`` -> name of grid to force.
    2. ``training.parameter_grid_size_threshold`` (int) -> n_train below threshold uses the
       small grid, otherwise the default grid. Defaults to ``DEFAULT_SMALL_GRID_THRESHOLD``.

    Returns the chosen grid and a label (``"default"`` or ``"small"``) for manifest output.
    """
    training_cfg = config.get("training", {}) or {}
    overrides = training_cfg.get("parameter_grid_by_dataset", {}) or {}
    forced = overrides.get(dataset_name)
    if isinstance(forced, str) and forced.strip():
        label = forced.strip().lower()
        if label == "small":
            return PARAMETER_GRID_SMALL, "small"
        if label == "default":
            return PARAMETER_GRID, "default"
    threshold = training_cfg.get("parameter_grid_size_threshold")
    try:
        threshold_value = int(threshold) if threshold is not None else DEFAULT_SMALL_GRID_THRESHOLD
    except (TypeError, ValueError):
        threshold_value = DEFAULT_SMALL_GRID_THRESHOLD
    if threshold_value > 0 and n_train < threshold_value:
        return PARAMETER_GRID_SMALL, "small"
    return PARAMETER_GRID, "default"

def load_training_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def discover_processed_datasets() -> list[Path]:
    return [
        path
        for path in sorted(PROCESSED_DATA_DIR.glob("*_clean.parquet"))
        if path.stem.replace("_clean", "").lower() in ELIGIBLE_TUNED_DATASETS
    ]


def build_processed_dataset_name(dataset_path: Path) -> str:
    return dataset_path.stem.replace("_clean", "").lower()


def load_saved_split_frames(dataset_name: str, cleaned_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_dir = SPLITS_DIR / dataset_name
    return reconstruct_split_frames(
        cleaned_df,
        dataset_dir / "train_ids.csv",
        dataset_dir / "val_ids.csv",
        dataset_dir / "test_ids.csv",
    )

def resolve_metrics_for_dataset(config: dict[str, Any], dataset_name: str, default_metrics: list[str]) -> list[str]:
    """Pick metric columns for a dataset from features.metrics_by_dataset or fall back to the global list."""
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


def _should_stratify(labels: pd.Series, use_stratify: bool = True) -> bool:
    counts = labels.value_counts(dropna=False)
    return bool(use_stratify and labels.nunique(dropna=True) > 1 and not counts.empty and counts.min() >= 2)


def split_native_jitline_frames(
    df: pd.DataFrame,
    val_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use the native train/(val/)test marker, carving validation off train when missing."""
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


def normalize_params(model_name: str, params: dict[str, Any], random_state: int) -> dict[str, Any]:
    normalized = dict(params)
    normalized.setdefault("random_state", random_state)
    if model_name == "rf":
        normalized.setdefault("n_jobs", -1)
    elif model_name == "xgb":
        normalized.setdefault("eval_metric", "logloss")
        normalized.setdefault("n_jobs", -1)
        normalized.setdefault("verbosity", 0)
    elif model_name == "lgbm":
        normalized.setdefault("verbosity", -1)
        normalized.setdefault("n_jobs", -1)
    return normalized


def build_model(model_name: str, params: dict[str, Any], random_state: int):
    base_model = get_model(model_name, random_state=random_state)
    tuned_model = clone(base_model)
    tuned_model.set_params(**normalize_params(model_name, params, random_state))
    return tuned_model


def _safe_float(value: Any, default: float = float("-inf")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def _safe_bool(value: Any) -> bool:
    return coerce_bool(value)

def _metric_or(row: dict[str, Any], primary: str, fallback: str) -> float:
    value = _safe_float(row.get(primary))
    return value if value != float("-inf") else _safe_float(row.get(fallback))

def _resolve_cv_folds(y_train: pd.Series, requested_folds: int) -> int:
    if requested_folds < 2 or y_train.nunique(dropna=True) < 2:
        return 0
    min_class_count = int(y_train.value_counts().min())
    return min(int(requested_folds), min_class_count)

def cross_validate_candidate(model: Any, X_train: pd.DataFrame, y_train: pd.Series, requested_folds: int) -> dict[str, Any]:
    """Evaluate one candidate with stratified CV on the training split."""
    cv_folds = _resolve_cv_folds(y_train, requested_folds)
    empty_record = {
        "cv_folds": int(cv_folds),
        "cv_mean_precision": float("nan"),
        "cv_std_precision": float("nan"),
        "cv_mean_recall": float("nan"),
        "cv_std_recall": float("nan"),
        "cv_mean_f1": float("nan"),
        "cv_std_f1": float("nan"),
        "cv_mean_auc": float("nan"),
        "cv_std_auc": float("nan"),
    }
    if cv_folds < 2:
        return empty_record

    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=splitter,
        scoring={
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "auc": "roc_auc",
        },
        n_jobs=1,
        error_score=np.nan,
    )
    record = {"cv_folds": int(cv_folds)}
    for metric in ["precision", "recall", "f1", "auc"]:
        values = scores.get(f"test_{metric}", np.asarray([], dtype=float))
        record[f"cv_mean_{metric}"] = float(np.nanmean(values)) if len(values) else float("nan")
        record[f"cv_std_{metric}"] = float(np.nanstd(values)) if len(values) else float("nan")
    return record


def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _metric_delta(left: float, right: float) -> float:
    if pd.isna(left) or pd.isna(right):
        return float("nan")
    return float(left - right)

def _gap_metrics(left_metrics: dict[str, float], right_metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_{metric}": _metric_delta(left_metrics[metric], right_metrics[metric])
        for metric in ["accuracy", "precision", "recall", "f1", "auc"]
    }


def select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No tuning candidates were generated")
    return sorted(
        candidates,
        key=lambda row: (
            _safe_bool(row.get("threshold_constraint_met", False)),
            _safe_float(row.get("val_recall")),
            _safe_float(row.get("val_f1")),
            _safe_float(row.get("val_auc")),
            _safe_float(row.get("val_precision")),
            -abs(_safe_float(row.get("gap_train_val_f1"), float("inf"))),
            -abs(_safe_float(row.get("gap_train_val_auc"), float("inf"))),
            _safe_float(row.get("cv_mean_recall")),
            _safe_float(row.get("cv_mean_f1")),
            _safe_float(row.get("cv_mean_auc")),
        ),
        reverse=True,
    )[0]


def make_candidate_record(
    *,
    dataset_name: str,
    model_name: str,
    candidate_index: int,
    params: dict[str, Any],
    feature_metadata: dict[str, Any],
    split_mode: str,
    random_seed: int,
    source_file: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    metrics: dict[str, float],
    model_path: Path,
    split_manifest_path: Path,
) -> dict[str, Any]:
    train_counts = y_train.value_counts().to_dict()
    val_counts = y_val.value_counts().to_dict()
    test_counts = y_test.value_counts().to_dict()
    return {
        "dataset_name": dataset_name,
        "model": model_name,
        "candidate_index": int(candidate_index),
        "split_mode": split_mode,
        "source_file": str(source_file),
        "split_manifest_path": str(split_manifest_path),
        "random_seed": int(random_seed),
        "num_train_rows": int(len(train_df)),
        "num_val_rows": int(len(val_df)),
        "num_test_rows": int(len(test_df)),
        "train_num_clean": int(train_counts.get(0, 0)),
        "train_num_defective": int(train_counts.get(1, 0)),
        "val_num_clean": int(val_counts.get(0, 0)),
        "val_num_defective": int(val_counts.get(1, 0)),
        "test_num_clean": int(test_counts.get(0, 0)),
        "test_num_defective": int(test_counts.get(1, 0)),
        "num_features": int(feature_metadata.get("num_features", 0)),
        "selected_metrics": ",".join(feature_metadata.get("selected_metrics", [])),
        "missing_metrics": ",".join(feature_metadata.get("missing_metrics", [])),
        "dropped_all_nan_metrics": ",".join(feature_metadata.get("dropped_all_nan_metrics", [])),
        "params_json": json.dumps(params, sort_keys=True),
        "model_path": str(model_path),
        **metrics,
    }


def run_tuned_metrics_training() -> None:
    ensure_project_dirs()
    TUNED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_training_config()
    random_seed = int(config.get("project", {}).get("random_seed", 42))
    default_metrics = list(config.get("features", {}).get("metrics", []))
    candidate_models = config.get("models", {}).get("candidates", ["rf", "xgb", "lgbm"])
    cv_folds = int(config.get("training", {}).get("cv_folds", DEFAULT_CV_FOLDS))
    val_size = float(config.get("split", {}).get("val_size", 0.1))
    set_global_seed(random_seed)

    processed_files = discover_processed_datasets()
    logger.info("Selected %s processed dataset(s) for tuned metrics training: %s", len(processed_files), ", ".join(build_processed_dataset_name(p) for p in processed_files))

    candidate_records: list[dict[str, Any]] = []
    best_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    ranking_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for dataset_path in processed_files:
        dataset_name = build_processed_dataset_name(dataset_path)
        try:
            df = read_parquet(dataset_path)
            if "label" not in df.columns:
                failures.append({"dataset_name": dataset_name, "stage": "dataset_validation", "error": "Label column is missing.", "source_file": str(dataset_path)})
                continue
            y_all = pd.to_numeric(df["label"], errors="coerce")
            if y_all.isna().any():
                failures.append({"dataset_name": dataset_name, "stage": "dataset_validation", "error": "Label column contains non-numeric values after preprocessing.", "source_file": str(dataset_path)})
                continue
            if y_all.nunique() < 2:
                failures.append({"dataset_name": dataset_name, "stage": "dataset_validation", "error": "Label column must contain at least two classes for tuning.", "source_file": str(dataset_path)})
                continue

            if dataset_name in JITLINE_DATASETS and "jitline_split" in df.columns:
                train_df, val_df, test_df = split_native_jitline_frames(df, val_size=val_size, random_seed=random_seed)
                split_mode = "jitline_native_split"
            else:
                train_df, val_df, test_df = load_saved_split_frames(dataset_name, df)
                split_mode = "saved_split"
            metrics = resolve_metrics_for_dataset(config, dataset_name, default_metrics)
            feature_spec = fit_metrics_feature_spec(train_df, metrics)
            X_train, y_train, train_feature_metadata = transform_metrics_training_frame(train_df, feature_spec)
            X_val, y_val, val_feature_metadata = transform_metrics_training_frame(val_df, feature_spec)
            X_test, y_test, test_feature_metadata = transform_metrics_training_frame(test_df, feature_spec)
            split_manifest_path = SPLITS_DIR / dataset_name / "manifest.json"

            if X_train.empty or X_train.shape[1] == 0:
                failures.append({"dataset_name": dataset_name, "stage": "feature_building", "error": "No usable metrics features found in the train split.", "source_file": str(dataset_path)})
                continue
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                failures.append({"dataset_name": dataset_name, "stage": "dataset_validation", "error": "Train and test labels must contain at least two classes for tuning.", "source_file": str(dataset_path)})
                continue

            threshold_strategy = resolve_threshold_strategy(config, dataset_name)
            n_train_rows = int(len(X_train))
            parameter_grid, parameter_grid_label = select_parameter_grid(config, dataset_name, n_train_rows)
            combined_feature_metadata = {
                "num_rows": int(len(df)),
                "num_features": int(train_feature_metadata.get("num_features", 0)),
                "selected_metrics": train_feature_metadata.get("selected_metrics", []),
                "missing_metrics": train_feature_metadata.get("missing_metrics", []),
                "dropped_all_nan_metrics": train_feature_metadata.get("dropped_all_nan_metrics", []),
            }

            for model_name in candidate_models:
                if model_name not in parameter_grid:
                    failures.append({"dataset_name": dataset_name, "model": model_name, "stage": "grid_configuration", "error": f"No tuning grid configured for model '{model_name}'.", "source_file": str(dataset_path), "parameter_grid_label": parameter_grid_label, "n_train_rows": n_train_rows})
                    continue

                model_candidates: list[dict[str, Any]] = []
                for candidate_index, params in enumerate(parameter_grid[model_name], start=1):
                    try:
                        model = build_model(model_name, params, random_seed)
                        model = configure_model_for_imbalance(model, y_train)
                        cv_metrics = cross_validate_candidate(model, X_train, y_train, requested_folds=cv_folds)
                        model.fit(X_train, y_train)
                        threshold_metadata = resolve_decision_threshold(
                            model,
                            X_val,
                            y_val,
                            precision_floor=THRESHOLD_PRECISION_FLOOR,
                            threshold_strategy=threshold_strategy,
                        )
                        decision_threshold = float(threshold_metadata["decision_threshold"])
                        train_metrics = evaluate_model(model, X_train, y_train, threshold=decision_threshold)
                        val_metrics = evaluate_model(model, X_val, y_val, threshold=decision_threshold)
                        test_metrics = evaluate_model(model, X_test, y_test, threshold=decision_threshold)
                        model_bundle = ModelBundle(
                            estimator=model,
                            feature_columns=list(X_train.columns),
                            decision_threshold=decision_threshold,
                            feature_family="metrics_only",
                            preprocessor=feature_spec,
                            metadata={
                                "dataset_name": dataset_name,
                                "model": model_name,
                                "candidate_index": int(candidate_index),
                                "feature_family": "metrics_only",
                                "decision_threshold": decision_threshold,
                                "selection_data_source": "validation_threshold",
                                "selection_strategy": "best_validation_threshold_candidate_cv_tiebreak",
                                "test_metrics_report_only": True,
                                **threshold_metadata,
                            },
                        )
                        model_path = TUNED_MODELS_DIR / dataset_name / f"{model_name}_candidate{candidate_index}.joblib"
                        save_model(model_bundle, model_path)
                        candidate_record = make_candidate_record(
                            dataset_name=dataset_name,
                            model_name=model_name,
                            candidate_index=candidate_index,
                            params=params,
                            feature_metadata=combined_feature_metadata,
                            split_mode=split_mode,
                            random_seed=random_seed,
                            source_file=dataset_path,
                            train_df=train_df,
                            val_df=val_df,
                            test_df=test_df,
                            y_train=y_train,
                            y_val=y_val,
                            y_test=y_test,
                            metrics={
                                **cv_metrics,
                                **threshold_metadata,
                                **_prefix_metrics(train_metrics, "train_"),
                                **_prefix_metrics(val_metrics, "val_"),
                                "accuracy": test_metrics["accuracy"],
                                "precision": test_metrics["precision"],
                                "recall": test_metrics["recall"],
                                "f1": test_metrics["f1"],
                                "auc": test_metrics["auc"],
                                **_gap_metrics(train_metrics, val_metrics, "gap_train_val"),
                                **_gap_metrics(train_metrics, test_metrics, "gap_train_test"),
                                "selection_data_source": "validation_threshold",
                                "selection_strategy": "best_validation_threshold_candidate_cv_tiebreak",
                                "test_metrics_report_only": True,
                            },
                            model_path=model_path,
                            split_manifest_path=split_manifest_path,
                        )
                        candidate_records.append(candidate_record)
                        model_candidates.append(candidate_record)
                    except Exception as exc:
                        failures.append({"dataset_name": dataset_name, "model": model_name, "candidate_index": candidate_index, "stage": "candidate_training", "error": str(exc), "source_file": str(dataset_path)})

                if not model_candidates:
                    continue

                best_candidate = select_best_candidate(model_candidates)
                selection_metric = f"val_recall_then_val_f1_precision_floor_{THRESHOLD_PRECISION_FLOOR:.2f}"
                best_records.append({**best_candidate, "selection_metric": selection_metric, "selection_strategy": "best_validation_threshold_candidate_cv_tiebreak"})
                summary_records.append(
                    {
                        "dataset_name": dataset_name,
                        "model": model_name,
                        "best_candidate_index": int(best_candidate["candidate_index"]),
                        "selection_metric": selection_metric,
                        "selection_strategy": "best_validation_threshold_candidate_cv_tiebreak",
                        "cv_folds": int(best_candidate.get("cv_folds", 0)),
                        "best_cv_precision": float(best_candidate.get("cv_mean_precision", float("nan"))),
                        "best_cv_recall": float(best_candidate.get("cv_mean_recall", float("nan"))),
                        "best_cv_f1": float(best_candidate.get("cv_mean_f1", float("nan"))),
                        "best_cv_auc": float(best_candidate.get("cv_mean_auc", float("nan"))),
                        "decision_threshold": float(best_candidate["decision_threshold"]),
                        "threshold_precision_floor": float(best_candidate["threshold_precision_floor"]),
                        "threshold_constraint_met": bool(best_candidate["threshold_constraint_met"]),
                        "threshold_selection_strategy": best_candidate["threshold_selection_strategy"],
                        "best_val_precision": float(best_candidate["val_precision"]),
                        "best_val_recall": float(best_candidate["val_recall"]),
                        "best_val_f1": float(best_candidate["val_f1"]),
                        "best_val_auc": float(best_candidate["val_auc"]),
                        "test_auc": float(best_candidate["auc"]),
                        "test_f1": float(best_candidate["f1"]),
                        "test_accuracy": float(best_candidate["accuracy"]),
                        "test_precision": float(best_candidate["precision"]),
                        "test_recall": float(best_candidate["recall"]),
                        "gap_train_val_f1": float(best_candidate["gap_train_val_f1"]),
                        "gap_train_val_auc": float(best_candidate["gap_train_val_auc"]),
                        "gap_train_test_f1": float(best_candidate["gap_train_test_f1"]),
                        "gap_train_test_auc": float(best_candidate["gap_train_test_auc"]),
                        "selection_data_source": best_candidate.get("selection_data_source", "validation_threshold"),
                        "test_metrics_report_only": bool(best_candidate.get("test_metrics_report_only", True)),
                        "split_mode": split_mode,
                        "source_file": str(dataset_path),
                        "split_manifest_path": str(split_manifest_path),
                        "random_seed": int(random_seed),
                        "params_json": best_candidate["params_json"],
                    }
                )
                ranking_records.append(
                    {
                        "dataset_name": dataset_name,
                        "model": model_name,
                        "auc": float(best_candidate["auc"]),
                        "f1": float(best_candidate["f1"]),
                        "precision": float(best_candidate["precision"]),
                        "recall": float(best_candidate["recall"]),
                        "accuracy": float(best_candidate["accuracy"]),
                        "decision_threshold": float(best_candidate["decision_threshold"]),
                        "threshold_precision_floor": float(best_candidate["threshold_precision_floor"]),
                        "threshold_constraint_met": bool(best_candidate["threshold_constraint_met"]),
                        "threshold_selection_strategy": best_candidate["threshold_selection_strategy"],
                        "cv_folds": int(best_candidate.get("cv_folds", 0)),
                        "cv_mean_precision": float(best_candidate.get("cv_mean_precision", float("nan"))),
                        "cv_mean_recall": float(best_candidate.get("cv_mean_recall", float("nan"))),
                        "cv_mean_f1": float(best_candidate.get("cv_mean_f1", float("nan"))),
                        "cv_mean_auc": float(best_candidate.get("cv_mean_auc", float("nan"))),
                        "train_f1": float(best_candidate["train_f1"]),
                        "val_precision": float(best_candidate["val_precision"]),
                        "val_recall": float(best_candidate["val_recall"]),
                        "val_f1": float(best_candidate["val_f1"]),
                        "val_auc": float(best_candidate["val_auc"]),
                        "gap_train_val_f1": float(best_candidate["gap_train_val_f1"]),
                        "gap_train_val_auc": float(best_candidate["gap_train_val_auc"]),
                        "gap_train_test_f1": float(best_candidate["gap_train_test_f1"]),
                        "gap_train_test_auc": float(best_candidate["gap_train_test_auc"]),
                        "selection_data_source": best_candidate.get("selection_data_source", "validation_threshold"),
                        "test_metrics_report_only": bool(best_candidate.get("test_metrics_report_only", True)),
                        "split_mode": split_mode,
                        "candidate_index": int(best_candidate["candidate_index"]),
                        "params_json": best_candidate["params_json"],
                        "model_path": best_candidate["model_path"],
                    }
                )

                logger.info(
                    "Selected best tuned %s model for %s with val_recall=%.4f, val_precision=%.4f, threshold=%.4f",
                    model_name,
                    dataset_name,
                    best_candidate["val_recall"],
                    best_candidate["val_precision"],
                    best_candidate["decision_threshold"],
                )

        except Exception as exc:
            failures.append({"dataset_name": dataset_name, "stage": "dataset_loading", "error": str(exc), "source_file": str(dataset_path)})

    candidate_results_df = pd.DataFrame(candidate_records)
    best_results_df = pd.DataFrame(best_records)
    summary_df = pd.DataFrame(summary_records)
    ranking_df = pd.DataFrame(ranking_records)
    failures_df = pd.DataFrame(failures)
    if failures_df.empty and len(failures_df.columns) == 0:
        failures_df = pd.DataFrame(columns=TUNING_FAILURE_COLUMNS)

    write_csv(candidate_results_df, TUNING_RESULTS_PATH)
    write_csv(best_results_df, TUNING_BEST_PATH)
    write_csv(summary_df, TUNING_SUMMARY_PATH)
    write_csv(ranking_df, TUNING_RANKING_PATH)
    write_csv(failures_df, TUNING_FAILURES_PATH)
    write_csv(summarize_results_table(ranking_df), RESULTS_TABLES_DIR / "metrics_tuned_summary_by_dataset.csv")
    write_csv(rank_models_by_dataset(ranking_df), RESULTS_TABLES_DIR / "metrics_tuned_ranked_by_dataset.csv")
    write_json(
        {
            "random_seed": random_seed,
            "default_metrics": list(default_metrics),
            "metrics_by_dataset": dict(config.get("features", {}).get("metrics_by_dataset", {}) or {}),
            "candidate_models": candidate_models,
            "parameter_grid": PARAMETER_GRID,
            "tuned_models_dir": str(TUNED_MODELS_DIR),
            "split_mode": "saved_split_or_jitline_native",
            "cv_folds": cv_folds,
            "selection_metric": f"val_recall_then_val_f1_precision_floor_{THRESHOLD_PRECISION_FLOOR:.2f}",
            "selection_strategy": "best_validation_threshold_candidate_cv_tiebreak",
            "selection_data_source": "validation_threshold",
            "threshold_precision_floor": THRESHOLD_PRECISION_FLOOR,
            "test_metrics_report_only": True,
        },
        TUNING_CONFIG_PATH,
    )

    logger.info("Saved tuning candidates to %s", TUNING_RESULTS_PATH)
    logger.info("Saved best tuned results to %s", TUNING_BEST_PATH)
    logger.info("Saved tuning summary to %s", TUNING_SUMMARY_PATH)
    logger.info("Saved tuning ranking to %s", TUNING_RANKING_PATH)
    logger.info("Saved tuning failures to %s", TUNING_FAILURES_PATH)


def main() -> None:
    run_tuned_metrics_training()


if __name__ == "__main__":
    main()
