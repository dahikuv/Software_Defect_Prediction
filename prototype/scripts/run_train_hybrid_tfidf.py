"""Run the hybrid TF-IDF baseline training experiments."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_project_config
from src.evaluation.compare import build_results_table
from src.features.commit_tfidf import build_tfidf_features, normalize_commit_text
from src.features.feature_merge import merge_feature_sets
from src.features.metrics_features import build_metrics_training_frame
from src.models.trainer import save_model, train_and_evaluate_model
from src.utils.io import read_csv, read_parquet, write_csv, write_parquet
from src.utils.logging import get_logger
from src.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_TABLES_DIR, ensure_project_dirs
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

PRIMARY_DATASET_NAMES = {"cm1", "jm1", "kc1", "pc1"}
GHPR_RAW_DIR = PROCESSED_DATA_DIR.parent / "raw" / "GHPR_dataset-master"
GHPR_PROCESSED_PATH = PROCESSED_DATA_DIR / "ghpr_hybrid_clean.parquet"
HYBRID_MODELS_DIR = MODELS_DIR / "hybrid_tfidf"
RESULTS_TABLE_PATH = RESULTS_TABLES_DIR / "hybrid_tfidf_results.csv"
TRAINING_FAILURES_PATH = RESULTS_TABLES_DIR / "hybrid_tfidf_failures.csv"
FEATURE_COVERAGE_PATH = RESULTS_TABLES_DIR / "hybrid_tfidf_feature_coverage.csv"
TFIDF_FEATURES_DIR = PROCESSED_DATA_DIR / "tfidf"

GHPR_METRIC_COLUMNS = [
    "cbo",
    "wmc",
    "dit",
    "rfc",
    "lcom",
    "totalMethods",
    "totalFields",
    "nosi",
    "loc",
    "returnQty",
    "loopQty",
    "comparisonsQty",
    "tryCatchQty",
    "parenthesizedExpsQty",
    "stringLiteralsQty",
    "numbersQty",
    "assignmentsQty",
    "mathOperationsQty",
    "variablesQty",
    "maxNestedBlocks",
    "uniqueWordsQty",
]


def load_training_config() -> dict[str, Any]:
    """Load training-related settings from the project config file."""
    return load_project_config()


def discover_processed_datasets() -> list[Path]:
    """Return cleaned parquet datasets ready for hybrid training."""
    paths = [
        path
        for path in sorted(PROCESSED_DATA_DIR.glob("*_clean.parquet"))
        if path.stem.replace("_clean", "") in PRIMARY_DATASET_NAMES
    ]
    if GHPR_PROCESSED_PATH.exists():
        paths.append(GHPR_PROCESSED_PATH)
    return paths


def build_processed_dataset_name(dataset_path: Path) -> str:
    """Convert a cleaned parquet path to the experiment dataset name."""
    return dataset_path.stem.replace("_clean", "")


def validate_training_configuration(metrics: list[str], model_candidates: list[str]) -> None:
    if not metrics:
        raise ValueError("The config must define at least one metric for hybrid training.")
    if not model_candidates:
        raise ValueError("The config must define at least one candidate model for training.")


def _compose_text(row: pd.Series) -> str:
    parts = []
    for column in ["COMMIT_DESCRIPTION", "PR_TITLE", "PR_DESCRIPTION", "DIFF_CODE", "PROJECT_DESCRIPTION", "PROJECT_LABEL"]:
        value = row.get(column, "")
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _normalize_sha(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def prepare_ghpr_hybrid_frame() -> pd.DataFrame:
    """Build a GHPR hybrid dataset with metrics, labels, and commit text."""
    baseline_path = GHPR_RAW_DIR / "baseline.csv"
    ghpr_path = GHPR_RAW_DIR / "ghprdata.csv"
    if not baseline_path.exists() or not ghpr_path.exists():
        raise FileNotFoundError("GHPR raw files baseline.csv and ghprdata.csv must exist")

    baseline_df = read_csv(baseline_path)
    ghpr_df = read_csv(ghpr_path)

    if "SHA" not in baseline_df.columns or "defect" not in baseline_df.columns:
        raise ValueError("GHPR baseline.csv must contain SHA and defect columns")
    if "SHA_FIXED" not in ghpr_df.columns or "SHA_BUG" not in ghpr_df.columns:
        raise ValueError("GHPR ghprdata.csv must contain SHA_FIXED and SHA_BUG columns")

    if "module_id" not in baseline_df.columns:
        baseline_df = baseline_df.copy()
        baseline_df["module_id"] = _normalize_sha(baseline_df["SHA"])
    baseline_df["label"] = baseline_df["defect"].astype(int)
    baseline_df["project_name"] = baseline_df.get("PROJECT_NAME", "GHPR")

    text_bug = ghpr_df.copy()
    text_bug["module_id"] = _normalize_sha(text_bug["SHA_BUG"])
    text_bug["text_role"] = "bug"
    text_fixed = ghpr_df.copy()
    text_fixed["module_id"] = _normalize_sha(text_fixed["SHA_FIXED"])
    text_fixed["text_role"] = "fixed"

    text_table = pd.concat([text_bug, text_fixed], ignore_index=True)
    text_table["commit_text"] = text_table.apply(_compose_text, axis=1)
    text_table = text_table[["module_id", "commit_text"]].copy()
    text_table = text_table[text_table["module_id"].astype(str).str.strip().ne("")]
    text_table = text_table.drop_duplicates(subset=["module_id"], keep="first")

    merged = baseline_df.merge(text_table, on="module_id", how="left")
    merged["commit_text"] = merged["commit_text"].fillna("").astype(str)
    merged["dataset_name"] = "ghpr"

    metric_columns = [col for col in GHPR_METRIC_COLUMNS if col in merged.columns]
    if not metric_columns:
        raise ValueError("GHPR dataset does not contain expected metric columns from baseline.csv")

    return merged


def build_hybrid_training_frame(df: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Build metrics + TF-IDF features for one cleaned dataset."""
    if "label" not in df.columns:
        raise ValueError("The input DataFrame must contain a 'label' column.")
    if "commit_text" not in df.columns:
        raise ValueError("The input DataFrame must contain a 'commit_text' column for TF-IDF training.")

    metrics_X, y, metric_metadata = build_metrics_training_frame(df, metrics)
    normalized_text = normalize_commit_text(df["commit_text"])
    has_commit_text = bool(normalized_text.str.split().str.len().fillna(0).gt(0).any())
    try:
        tfidf_model, tfidf_X = build_tfidf_features(normalized_text)
    except ValueError as exc:
        raise ValueError(f"No usable commit_text for TF-IDF features: {exc}") from exc

    tfidf_path = TFIDF_FEATURES_DIR / f"{df.iloc[0].get('project_name', 'unknown_project')}_tfidf.parquet"
    write_parquet(tfidf_X, tfidf_path)

    X = merge_feature_sets(metrics_X, text_df=tfidf_X)
    metadata = {
        **metric_metadata,
        "tfidf_num_features": int(tfidf_X.shape[1]),
        "tfidf_vocabulary_size": int(len(getattr(tfidf_model, "vocabulary_", {}))),
        "num_rows": int(len(df)),
        "has_commit_text": has_commit_text,
    }
    return X, y, metadata


def stringify_label_distribution(y: Any) -> str:
    counts = y.value_counts().to_dict()
    ordered_items = sorted((int(label), int(count)) for label, count in counts.items())
    return ",".join(f"{label}:{count}" for label, count in ordered_items)


def build_split_distribution_record(y_train: Any, y_test: Any) -> dict[str, int]:
    train_counts = y_train.value_counts().to_dict()
    test_counts = y_test.value_counts().to_dict()
    return {
        "train_num_clean": int(train_counts.get(0, 0)),
        "train_num_defective": int(train_counts.get(1, 0)),
        "test_num_clean": int(test_counts.get(0, 0)),
        "test_num_defective": int(test_counts.get(1, 0)),
    }


def build_split_summary_record(dataset_name: str, y_train: Any, y_test: Any) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "train_label_distribution": stringify_label_distribution(y_train),
        "test_label_distribution": stringify_label_distribution(y_test),
        **build_split_distribution_record(y_train, y_test),
    }


def build_feature_coverage_record(dataset_name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    label_distribution = metadata.get("label_distribution", {})
    return {
        "dataset_name": dataset_name,
        "num_rows": int(metadata.get("num_rows", 0)),
        "num_metrics_features": int(metadata.get("num_features", 0)),
        "num_tfidf_features": int(metadata.get("tfidf_num_features", 0)),
        "tfidf_vocabulary_size": int(metadata.get("tfidf_vocabulary_size", 0)),
        "selected_metrics": ",".join(metadata.get("selected_metrics", [])),
        "missing_metrics": ",".join(metadata.get("missing_metrics", [])),
        "dropped_all_nan_metrics": ",".join(metadata.get("dropped_all_nan_metrics", [])),
        "num_clean": int(label_distribution.get(0, 0)),
        "num_defective": int(label_distribution.get(1, 0)),
        "has_commit_text": bool(metadata.get("has_commit_text", False)),
    }


def log_training_configuration(metrics: list[str], model_candidates: list[str], random_seed: int, test_size: float, use_stratify: bool) -> None:
    logger.info(
        "Hybrid training configuration -> seed=%s, test_size=%s, stratify=%s, models=%s",
        random_seed,
        test_size,
        use_stratify,
        ", ".join(model_candidates),
    )
    logger.info("Configured metrics (%s): %s", len(metrics), ", ".join(metrics))


def should_use_stratify(y: Any, use_stratify: bool) -> bool:
    class_counts = y.value_counts()
    return bool(use_stratify and y.nunique() > 1 and not class_counts.empty and class_counts.min() >= 2)


def split_dataset(X: Any, y: Any, test_size: float, random_seed: int, stratify_labels: Any) -> tuple[Any, Any, Any, Any]:
    return train_test_split(X, y, test_size=test_size, random_state=random_seed, stratify=stratify_labels)


def run_hybrid_tfidf_training() -> None:
    """Execute the full hybrid metrics + TF-IDF training flow."""
    ensure_project_dirs()
    TFIDF_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    HYBRID_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_training_config()
    random_seed = int(config.get("project", {}).get("random_seed", 42))
    metrics = config.get("features", {}).get("metrics", [])
    model_candidates = config.get("models", {}).get("candidates", ["rf"])
    test_size = float(config.get("split", {}).get("test_size", 0.2))
    use_stratify = bool(config.get("split", {}).get("stratify", True))

    validate_training_configuration(metrics, model_candidates)
    set_global_seed(random_seed)
    log_training_configuration(metrics, model_candidates, random_seed, test_size, use_stratify)

    processed_files = discover_processed_datasets()
    logger.info("Selected %s dataset(s) for hybrid training.", len(processed_files))

    all_results: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    all_feature_coverage_records: list[dict[str, Any]] = []

    for dataset_path in processed_files:
        dataset_name = build_processed_dataset_name(dataset_path)
        try:
            if dataset_path == GHPR_PROCESSED_PATH:
                df = prepare_ghpr_hybrid_frame()
                feature_source = "ghpr_raw"
            else:
                df = read_parquet(dataset_path)
                feature_source = str(dataset_path)

            X, y, feature_metadata = build_hybrid_training_frame(df, metrics)
            coverage_record = {
                **build_feature_coverage_record(dataset_name, feature_metadata),
                "configured_metrics": ",".join(metrics),
                "configured_models": ",".join(model_candidates),
                "random_seed": int(random_seed),
                "test_size": float(test_size),
                "stratify_enabled": bool(use_stratify),
                "feature_mode": "metrics+tfidf",
                "source_file": feature_source,
            }
            all_feature_coverage_records.append(coverage_record)

            if X.empty or X.shape[1] == 0:
                all_failures.append({
                    "dataset_name": dataset_name,
                    "stage": "tfidf_feature_building",
                    "error": "No usable commit_text for TF-IDF features or no hybrid features remained after preprocessing.",
                    "source_file": feature_source,
                })
                continue
            if y.nunique() < 2:
                all_failures.append({
                    "dataset_name": dataset_name,
                    "stage": "dataset_validation",
                    "error": "Label column must contain at least two classes for training.",
                    "source_file": feature_source,
                })
                continue

            can_stratify = should_use_stratify(y, use_stratify)
            stratify_labels = y if can_stratify else None
            try:
                X_train, X_test, y_train, y_test = split_dataset(X, y, test_size, random_seed, stratify_labels)
            except Exception as exc:
                all_failures.append({
                    "dataset_name": dataset_name,
                    "stage": "train_test_split",
                    "error": str(exc),
                    "stratified_split": bool(can_stratify),
                    "source_file": feature_source,
                })
                continue

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
                    model_path = HYBRID_MODELS_DIR / f"{model_name}_{dataset_name}.joblib"
                    save_model(model, model_path)
                    result.update(
                        {
                            "source_file": feature_source,
                            "random_seed": int(random_seed),
                            "test_size": float(test_size),
                            "stratified_split": bool(can_stratify),
                            "model_path": str(model_path),
                            "feature_mode": "metrics+tfidf",
                            "feature_family": result.get("feature_family", "metrics_plus_commit_text"),
                            "feature_set": result.get("feature_set", result.get("feature_family", "metrics_plus_commit_text")),
                            "uses_commit_text": True,
                            "artifact_stage": "training",
                            "artifact_schema_version": "paper-v1",
                            "artifact_group_key": f"{dataset_name}::{model_name}",
                            "artifact_id": f"{dataset_name}::{model_name}::training",
                            "source_results_table": str(RESULTS_TABLE_PATH),
                            **build_split_summary_record(dataset_name, y_train, y_test),
                        }
                    )
                    all_results.append(result)
                except Exception as exc:
                    all_failures.append({
                        "dataset_name": dataset_name,
                        "model": model_name,
                        "stage": "model_training",
                        "error": str(exc),
                        "source_file": feature_source,
                    })
        except Exception as exc:
            all_failures.append({"dataset_name": dataset_name, "stage": "dataset_loading", "error": str(exc), "source_file": str(dataset_path)})

    write_csv(build_results_table(all_feature_coverage_records), FEATURE_COVERAGE_PATH)
    write_csv(build_results_table(all_results), RESULTS_TABLE_PATH)
    write_csv(build_results_table(all_failures), TRAINING_FAILURES_PATH)
    logger.info("Saved hybrid feature coverage to %s", FEATURE_COVERAGE_PATH)
    logger.info("Saved hybrid results table to %s", RESULTS_TABLE_PATH)
    logger.info("Saved hybrid training failures table to %s", TRAINING_FAILURES_PATH)
    logger.info(
        "Hybrid TF-IDF training completed: %s successful run(s), %s failure record(s).",
        len(all_results),
        len(all_failures),
    )


def main() -> None:
    run_hybrid_tfidf_training()


if __name__ == "__main__":
    main()
