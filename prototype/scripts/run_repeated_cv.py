"""Repeated 10-fold stratified cross-validation for all datasets and models."""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from src.evaluation.metrics import compute_classification_metrics
from src.features.metrics_features import build_metrics_training_frame
from src.models.registry import get_model
from src.models.trainer import configure_model_for_imbalance
from src.utils.io import read_csv, read_parquet, write_csv
from src.utils.logging import get_logger
from src.utils.paths import PROCESSED_DATA_DIR, RESULTS_TABLES_DIR, SPLITS_DIR, ensure_project_dirs
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

JITLINE_DATASETS = {"openstack", "qt", "jitfine"}
DEFAULT_METRICS = ["loc", "v(g)", "ev(g)", "iv(g)", "branchCount"]

OUTPUT_TABLE = RESULTS_TABLES_DIR / "repeated_cv_results.csv"

DATASETS = {
    "cm1": {"parquet": "cm1_clean.parquet", "split": "cm1"},
    "jm1": {"parquet": "jm1_clean.parquet", "split": "jm1"},
    "kc1": {"parquet": "kc1_clean.parquet", "split": "kc1"},
    "pc1": {"parquet": "pc1_clean.parquet", "split": "pc1"},
    "jitfine": {"parquet": "jitfine_clean.parquet", "split": "jitfine"},
    "openstack": {"parquet": "openstack_clean.parquet", "split": "openstack"},
    "qt": {"parquet": "qt_clean.parquet", "split": "qt"},
}

def get_feature_columns(df):
    exclude = {"label", "module_id", "project_name", "dataset_name", "commit_text", "jitline_split",
               "classification", "fix", "is_buggy_commit", "author_date", "author_name", "author_email",
               "parent_hashes", "commit_hash", "fileschanged", "author_date_unix_timestamp"}
    return [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64', 'int32', 'bool']]


def load_dataset_data(ds_name, info):
    parquet_path = PROCESSED_DATA_DIR / info["parquet"]
    if not parquet_path.exists():
        # Fallback to split row files
        split_dir = SPLITS_DIR / info["split"]
        train_rows = split_dir / "train_rows.csv"
        test_rows = split_dir / "test_rows.csv"
        if train_rows.exists() and test_rows.exists():
            train = read_csv(train_rows)
            test = read_csv(test_rows)
            return pd.concat([train, test], ignore_index=True)
        return None
    return read_parquet(parquet_path)


def run_repeated_cv():
    ensure_project_dirs()
    set_global_seed(42)

    all_results = []

    for ds_name, info in DATASETS.items():
        df = load_dataset_data(ds_name, info)
        if df is None:
            logger.warning("Skipping %s: data not found", ds_name)
            continue

        df = df.fillna(0)
        feat_cols = get_feature_columns(df)
        X = df[feat_cols].values
        y = df["label"].astype(int).values

        logger.info("[%s] n=%d, features=%d, defective=%d (%.1f%%)",
                    ds_name, len(df), len(feat_cols), y.sum(), 100*y.mean())

        rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

        for model_name in ["rf", "xgb", "lgbm"]:
            fold_results = []
            for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = get_model(model_name, random_state=42)
                model = configure_model_for_imbalance(model, pd.Series(y_train))
                model.fit(X_train, y_train)

                y_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_proba >= 0.5).astype(int)
                metrics = compute_classification_metrics(pd.Series(y_test), pd.Series(y_pred), pd.Series(y_proba))

                fold_results.append({
                    "dataset_name": ds_name,
                    "model": model_name,
                    "fold": fold_idx,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "auc": metrics["auc"],
                })

            fold_df = pd.DataFrame(fold_results)
            summary = {
                "dataset_name": ds_name,
                "model": model_name,
                "n_folds": len(fold_results),
                "accuracy_mean": fold_df["accuracy"].mean(),
                "accuracy_std": fold_df["accuracy"].std(),
                "precision_mean": fold_df["precision"].mean(),
                "precision_std": fold_df["precision"].std(),
                "recall_mean": fold_df["recall"].mean(),
                "recall_std": fold_df["recall"].std(),
                "f1_mean": fold_df["f1"].mean(),
                "f1_std": fold_df["f1"].std(),
                "auc_mean": fold_df["auc"].mean(),
                "auc_std": fold_df["auc"].std(),
            }
            all_results.append(summary)
            logger.info("  %s %s: AUC=%.3f+-%.3f Recall=%.3f+-%.3f F1=%.3f+-%.3f",
                        ds_name, model_name,
                        summary["auc_mean"], summary["auc_std"],
                        summary["recall_mean"], summary["recall_std"],
                        summary["f1_mean"], summary["f1_std"])

    results_df = pd.DataFrame(all_results)
    write_csv(results_df, OUTPUT_TABLE)
    logger.info("Saved repeated CV results to %s", OUTPUT_TABLE)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    run_repeated_cv()
