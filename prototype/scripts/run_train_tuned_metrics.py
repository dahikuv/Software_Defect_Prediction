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
from sklearn.base import clone

from src.data.split import reconstruct_split_frames
from src.evaluation.compare import build_results_table, rank_models_by_dataset, summarize_results_table
from src.features.metrics_features import build_metrics_training_frame
from src.models.registry import get_model
from src.models.trainer import evaluate_model, save_model
from src.utils.io import read_parquet, write_csv, write_json
from src.utils.logging import get_logger
from src.utils.paths import CONFIG_PATH, MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_TABLES_DIR, SPLITS_DIR, ensure_project_dirs
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

PRIMARY_DATASET_NAMES = {"cm1", "jm1", "kc1", "pc1"}
TUNED_MODELS_DIR = MODELS_DIR / "metrics_tuned"
TUNING_RESULTS_PATH = RESULTS_TABLES_DIR / "metrics_tuned_results.csv"
TUNING_CANDIDATES_PATH = RESULTS_TABLES_DIR / "metrics_tuning_candidates.csv"
TUNING_BEST_PATH = RESULTS_TABLES_DIR / "metrics_tuned_best.csv"
TUNING_SUMMARY_PATH = RESULTS_TABLES_DIR / "metrics_tuned_summary.csv"
TUNING_RANKING_PATH = RESULTS_TABLES_DIR / "metrics_tuned_ranking.csv"
TUNING_FAILURES_PATH = RESULTS_TABLES_DIR / "metrics_tuned_failures.csv"
TUNING_CONFIG_PATH = RESULTS_TABLES_DIR / "metrics_tuned_config.json"


PARAMETER_GRID: dict[str, list[dict[str, Any]]] = {
    "rf": [
        {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": 12, "min_samples_split": 4, "min_samples_leaf": 1},
        {"n_estimators": 400, "max_depth": 18, "min_samples_split": 2, "min_samples_leaf": 2},
    ],
    "xgb": [
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
    ],
    "lgbm": [
        {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 300, "num_leaves": 63, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 400, "num_leaves": 31, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
    ],
}


def load_training_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def discover_processed_datasets() -> list[Path]:
    return [
        path
        for path in sorted(PROCESSED_DATA_DIR.glob("*_clean.parquet"))
        if path.stem.replace("_clean", "") in PRIMARY_DATASET_NAMES
    ]


def build_processed_dataset_name(dataset_path: Path) -> str:
    return dataset_path.stem.replace("_clean", "")


def load_saved_split_frames(dataset_name: str, cleaned_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_dir = SPLITS_DIR / dataset_name
    return reconstruct_split_frames(
        cleaned_df,
        dataset_dir / "train_ids.csv",
        dataset_dir / "val_ids.csv",
        dataset_dir / "test_ids.csv",
    )


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


def select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No tuning candidates were generated")
    return sorted(
        candidates,
        key=lambda row: (
            row.get("auc", float("-inf")),
            row.get("f1", float("-inf")),
            row.get("recall", float("-inf")),
            row.get("precision", float("-inf")),
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
    metrics = config.get("features", {}).get("metrics", [])
    candidate_models = config.get("models", {}).get("candidates", ["rf", "xgb", "lgbm"])
    set_global_seed(random_seed)

    processed_files = discover_processed_datasets()
    logger.info("Selected %s primary processed dataset(s) for tuned metrics training.", len(processed_files))

    candidate_records: list[dict[str, Any]] = []
    best_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    ranking_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for dataset_path in processed_files:
        dataset_name = build_processed_dataset_name(dataset_path)
        try:
            df = read_parquet(dataset_path)
            X_all, y_all, feature_metadata = build_metrics_training_frame(df, metrics)
            if X_all.empty or X_all.shape[1] == 0:
                failures.append({"dataset_name": dataset_name, "stage": "feature_building", "error": "No usable metrics features found after preprocessing.", "source_file": str(dataset_path)})
                continue
            if y_all.nunique() < 2:
                failures.append({"dataset_name": dataset_name, "stage": "dataset_validation", "error": "Label column must contain at least two classes for tuning.", "source_file": str(dataset_path)})
                continue

            train_df, val_df, test_df = load_saved_split_frames(dataset_name, df)
            X_train, y_train, train_feature_metadata = build_metrics_training_frame(train_df, metrics)
            X_val, y_val, val_feature_metadata = build_metrics_training_frame(val_df, metrics)
            X_test, y_test, test_feature_metadata = build_metrics_training_frame(test_df, metrics)
            split_manifest_path = SPLITS_DIR / dataset_name / "split_manifest.json"

            split_mode = "saved_split"
            combined_feature_metadata = {
                "num_rows": int(feature_metadata.get("num_rows", 0)),
                "num_features": int(feature_metadata.get("num_features", 0)),
                "selected_metrics": feature_metadata.get("selected_metrics", []),
                "missing_metrics": feature_metadata.get("missing_metrics", []),
                "dropped_all_nan_metrics": feature_metadata.get("dropped_all_nan_metrics", []),
            }

            for model_name in candidate_models:
                if model_name not in PARAMETER_GRID:
                    failures.append({"dataset_name": dataset_name, "model": model_name, "stage": "grid_configuration", "error": f"No tuning grid configured for model '{model_name}'.", "source_file": str(dataset_path)})
                    continue

                model_candidates: list[dict[str, Any]] = []
                for candidate_index, params in enumerate(PARAMETER_GRID[model_name], start=1):
                    try:
                        model = build_model(model_name, params, random_seed)
                        model.fit(X_train, y_train)
                        val_metrics = evaluate_model(model, X_val, y_val)
                        test_metrics = evaluate_model(model, X_test, y_test)
                        model_path = TUNED_MODELS_DIR / dataset_name / f"{model_name}_candidate{candidate_index}.joblib"
                        save_model(model, model_path)
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
                                "val_accuracy": val_metrics["accuracy"],
                                "val_precision": val_metrics["precision"],
                                "val_recall": val_metrics["recall"],
                                "val_f1": val_metrics["f1"],
                                "val_auc": val_metrics["auc"],
                                "accuracy": test_metrics["accuracy"],
                                "precision": test_metrics["precision"],
                                "recall": test_metrics["recall"],
                                "f1": test_metrics["f1"],
                                "auc": test_metrics["auc"],
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
                best_records.append({**best_candidate, "selection_metric": "val_auc", "selection_strategy": "best_validation_candidate"})
                summary_records.append(
                    {
                        "dataset_name": dataset_name,
                        "model": model_name,
                        "best_candidate_index": int(best_candidate["candidate_index"]),
                        "selection_metric": "val_auc",
                        "selection_strategy": "best_validation_candidate",
                        "best_val_auc": float(best_candidate["val_auc"]),
                        "best_val_f1": float(best_candidate["val_f1"]),
                        "test_auc": float(best_candidate["auc"]),
                        "test_f1": float(best_candidate["f1"]),
                        "test_accuracy": float(best_candidate["accuracy"]),
                        "test_precision": float(best_candidate["precision"]),
                        "test_recall": float(best_candidate["recall"]),
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
                        "split_mode": split_mode,
                        "candidate_index": int(best_candidate["candidate_index"]),
                        "params_json": best_candidate["params_json"],
                        "model_path": best_candidate["model_path"],
                    }
                )

                logger.info("Selected best tuned %s model for %s with val_auc=%.4f", model_name, dataset_name, best_candidate["val_auc"])

        except Exception as exc:
            failures.append({"dataset_name": dataset_name, "stage": "dataset_loading", "error": str(exc), "source_file": str(dataset_path)})

    candidate_results_df = pd.DataFrame(candidate_records)
    best_results_df = pd.DataFrame(best_records)
    summary_df = pd.DataFrame(summary_records)
    ranking_df = pd.DataFrame(ranking_records)
    failures_df = pd.DataFrame(failures)

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
            "metrics": metrics,
            "candidate_models": candidate_models,
            "parameter_grid": PARAMETER_GRID,
            "tuned_models_dir": str(TUNED_MODELS_DIR),
            "split_mode": "saved_split",
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
