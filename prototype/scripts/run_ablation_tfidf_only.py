"""Ablation study: TF-IDF-only (commit messages without metrics) on all JIT datasets."""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.evaluation.metrics import compute_classification_metrics, select_decision_threshold
from src.features.commit_tfidf import normalize_commit_text
from src.models.registry import get_model
from src.models.trainer import configure_model_for_imbalance
from src.utils.io import read_parquet, write_csv
from src.utils.logging import get_logger
from src.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_TABLES_DIR, ensure_project_dirs
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

JIT_DATASETS = {
    "jitfine": PROCESSED_DATA_DIR / "jitfine_clean.parquet",
    "openstack": PROCESSED_DATA_DIR / "openstack_clean.parquet",
    "qt": PROCESSED_DATA_DIR / "qt_clean.parquet",
}
JITLINE_DATASETS = {"openstack", "qt"}
OUTPUT_TABLE = RESULTS_TABLES_DIR / "ablation_tfidf_only.csv"
ABLATION_MODELS_DIR = MODELS_DIR / "ablation_tfidf_only"


def run_ablation():
    ensure_project_dirs()
    ABLATION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    set_global_seed(42)

    all_results = []

    for dataset_name, data_path in JIT_DATASETS.items():
        if not data_path.exists():
            logger.error("Data not found: %s", data_path)
            continue

        df = read_parquet(data_path)
        logger.info("Loaded %s: %d rows", dataset_name, len(df))

        # Split
        if dataset_name in JITLINE_DATASETS:
            train_df = df[df["jitline_split"] == "train"].copy()
            val_df = df[df["jitline_split"] == "val"].copy() if "val" in df["jitline_split"].values else pd.DataFrame()
            test_df = df[df["jitline_split"] == "test"].copy()
            # If no val split, use last 20% of train as val
            if val_df.empty:
                n_val = int(len(train_df) * 0.2)
                val_df = train_df.iloc[-n_val:].copy()
                train_df = train_df.iloc[:-n_val].copy()
        else:
            train_df = df[df["jitline_split"] == "train"].copy()
            val_df = df[df["jitline_split"] == "val"].copy()
            test_df = df[df["jitline_split"] == "test"].copy()

        logger.info("Split: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

        # Fit TF-IDF on training commit_text only
        text_train = normalize_commit_text(train_df["commit_text"])
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        vectorizer.fit(text_train)
        feature_names = list(vectorizer.get_feature_names_out())
        logger.info("TF-IDF vocabulary: %d features", len(feature_names))

        # Transform splits
        def tfidf_transform(frame):
            text = normalize_commit_text(frame["commit_text"])
            matrix = vectorizer.transform(text)
            return pd.DataFrame(
                matrix.toarray().astype("float32"),
                columns=[f"commit_{name}" for name in feature_names],
                index=frame.index,
            )

        X_train = tfidf_transform(train_df)
        y_train = train_df["label"].astype(int)
        X_val = tfidf_transform(val_df)
        y_val = val_df["label"].astype(int)
        X_test = tfidf_transform(test_df)
        y_test = test_df["label"].astype(int)

        # Train all 3 models
        for model_name in ["rf", "xgb", "lgbm"]:
            logger.info("[%s] Training TF-IDF-only %s ...", dataset_name, model_name)
            model = get_model(model_name, random_state=42)
            model = configure_model_for_imbalance(model, y_train)
            model.fit(X_train, y_train)

            # Threshold selection on validation
            threshold_info = select_decision_threshold(
                y_val, model.predict_proba(X_val)[:, 1],
                strategy="recall_with_precision_floor",
                precision_floor=0.30,
            )
            threshold = threshold_info["decision_threshold"]

            # Evaluate on test
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)
            metrics = compute_classification_metrics(y_test, y_pred, y_proba)

            result = {
                "dataset_name": dataset_name,
                "model": model_name,
                "feature_family": "tfidf_only",
                "num_features": X_train.shape[1],
                "decision_threshold": threshold,
                **threshold_info,
                **metrics,
            }
            all_results.append(result)
            logger.info("  %s: AUC=%.3f Recall=%.3f F1=%.3f", model_name, metrics["auc"], metrics["recall"], metrics["f1"])

    results_df = pd.DataFrame(all_results)
    write_csv(results_df, OUTPUT_TABLE)
    logger.info("Saved ablation results to %s", OUTPUT_TABLE)
    print(results_df[["dataset_name", "model", "accuracy", "precision", "recall", "f1", "auc", "decision_threshold"]].to_string(index=False))


if __name__ == "__main__":
    run_ablation()
