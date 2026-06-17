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
from src.data.split import reconstruct_split_frames
from src.evaluation.compare import build_results_table
from src.features.hybrid_tfidf import build_hybrid_tfidf_training_frame, fit_hybrid_tfidf_feature_spec
from src.models.trainer import save_model, train_and_evaluate_model
from src.utils.io import read_csv, read_parquet, write_csv
from src.utils.logging import get_logger
from src.utils.paths import MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_TABLES_DIR, SPLITS_DIR, ensure_project_dirs
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

PRIMARY_DATASET_NAMES = {"cm1", "jm1", "kc1", "pc1"}
JITLINE_DATASETS = {"openstack", "qt", "jitfine"}
GHPR_RAW_DIR = PROCESSED_DATA_DIR.parent / "raw" / "GHPR_dataset-master"
GHPR_PROCESSED_PATH = PROCESSED_DATA_DIR / "ghpr_hybrid_clean.parquet"
GHPR_PAIR_POLICY = "drop_conflicting_fix_sha"
HYBRID_MODELS_DIR = MODELS_DIR / "hybrid_tfidf"
RESULTS_TABLE_PATH = RESULTS_TABLES_DIR / "hybrid_tfidf_results.csv"
TRAINING_FAILURES_PATH = RESULTS_TABLES_DIR / "hybrid_tfidf_failures.csv"
FEATURE_COVERAGE_PATH = RESULTS_TABLES_DIR / "hybrid_tfidf_feature_coverage.csv"
HYBRID_RESULTS_COLUMNS = [
    "dataset_name",
    "model",
    "feature_family",
    "feature_set",
    "uses_commit_text",
    "text_feature_column",
    "tfidf_num_features",
    "tfidf_vocabulary_size",
    "num_train_rows",
    "num_val_rows",
    "num_test_rows",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "model_path",
]

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
    eligible = PRIMARY_DATASET_NAMES | JITLINE_DATASETS
    paths = [
        path
        for path in sorted(PROCESSED_DATA_DIR.glob("*_clean.parquet"))
        if path.stem.replace("_clean", "").lower() in eligible
    ]
    if GHPR_PROCESSED_PATH.exists() or ((GHPR_RAW_DIR / "baseline.csv").exists() and (GHPR_RAW_DIR / "ghprdata.csv").exists()):
        paths.append(GHPR_PROCESSED_PATH)
    return paths

def build_processed_dataset_name(dataset_path: Path) -> str:
    """Convert a cleaned parquet path to the experiment dataset name."""
    if dataset_path == GHPR_PROCESSED_PATH:
        return "ghpr"
    return dataset_path.stem.replace("_clean", "").lower()

def resolve_threshold_strategy(config: dict[str, Any], dataset_name: str) -> str:
    """Resolve the validation-threshold strategy from config (with per-dataset override)."""
    training_cfg = config.get("training", {}) or {}
    overrides = training_cfg.get("threshold_strategy_by_dataset", {}) or {}
    override = overrides.get(dataset_name)
    if override:
        return str(override)
    default = training_cfg.get("threshold_strategy", "recall_with_precision_floor")
    return str(default)

def resolve_metrics_for_dataset(config: dict[str, Any], dataset_name: str, default_metrics: list[str]) -> list[str]:
    """Pick metric columns for a dataset from `features.metrics_by_dataset` or fall back to the global list."""
    if dataset_name == "ghpr":
        return list(GHPR_METRIC_COLUMNS)
    by_dataset = config.get("features", {}).get("metrics_by_dataset", {}) or {}
    override = by_dataset.get(dataset_name)
    if override:
        return [str(metric) for metric in override]
    return list(default_metrics)

def validate_training_configuration(metrics: list[str], model_candidates: list[str]) -> None:
    if not metrics:
        raise ValueError("The config must define at least one metric for hybrid training.")
    if not model_candidates:
        raise ValueError("The config must define at least one candidate model for training.")

# GHPR commit-text composition. We deliberately exclude DIFF_CODE here: source
# diffs share filenames, identifiers, and tokens between the buggy fix and its
# follow-up commit, which leaks signal into TF-IDF and inflates the hybrid
# branch's metrics. PROJECT_DESCRIPTION/PROJECT_LABEL also carry per-project
# constants that act as project IDs and are excluded for the same reason. The
# diff content is preserved separately via _compose_diff_text so a future
# diff-only feature branch can use it without contaminating commit_text.
GHPR_COMMIT_TEXT_COLUMNS = ("COMMIT_DESCRIPTION", "PR_TITLE", "PR_DESCRIPTION")
GHPR_DIFF_TEXT_COLUMNS = ("DIFF_CODE",)

def _compose_text(row: pd.Series) -> str:
    parts: list[str] = []
    for column in GHPR_COMMIT_TEXT_COLUMNS:
        value = row.get(column, "")
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()

def _compose_diff_text(row: pd.Series) -> str:
    parts: list[str] = []
    for column in GHPR_DIFF_TEXT_COLUMNS:
        value = row.get(column, "")
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()

def _normalize_sha(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()

def _ghpr_pair_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """Return pair-policy metadata carried by the GHPR adapter."""
    return {
        key: df.attrs[key]
        for key in [
            "ghpr_pair_policy",
            "ghpr_rows_before_pair_policy",
            "ghpr_rows_after_pair_policy",
            "ghpr_conflicting_fix_sha_count",
            "ghpr_conflicting_rows_dropped",
        ]
        if key in df.attrs
    }

def prepare_ghpr_hybrid_frame() -> pd.DataFrame:
    """Build a GHPR hybrid dataset with metrics, labels, and commit text."""
    baseline_path = GHPR_RAW_DIR / "baseline.csv"
    ghpr_path = GHPR_RAW_DIR / "ghprdata.csv"
    if not baseline_path.exists() or not ghpr_path.exists():
        raise FileNotFoundError("GHPR raw files baseline.csv and ghprdata.csv must exist")

    GHPR_GHPRDATA_COLUMNS = [
        "PROJECT_NAME",
        "PROJECT_OWNER",
        "PROJECT_DESCRIPTION",
        "PROJECT_LABEL",
        "PROJECT_LANGUAGE",
        "SHA_FIXED",
        "SHA_BUG",
        "DIFF_CODE",
        "COMMIT_DESCRIPTION",
        "COMMIT_TIME",
        "OLD_CONTENT",
        "NEW_CONTENT",
        "OLD_PATH",
        "NEW_PATH",
        "PR_TITLE",
        "PR_DESCRIPTION",
    ]
    baseline_df = read_csv(baseline_path)

    ghpr_probe = pd.read_csv(ghpr_path, nrows=1, header=None, low_memory=False, encoding="utf-8")
    if ghpr_probe.shape[1] != len(GHPR_GHPRDATA_COLUMNS):
        raise ValueError(
            f"GHPR ghprdata.csv has {ghpr_probe.shape[1]} columns; expected {len(GHPR_GHPRDATA_COLUMNS)} from the README schema."
        )
    ghpr_df = pd.read_csv(
        ghpr_path,
        header=None,
        names=GHPR_GHPRDATA_COLUMNS,
        dtype=str,
        low_memory=False,
        encoding="utf-8",
        na_filter=False,
    )

    if "SHA" not in baseline_df.columns or "defect" not in baseline_df.columns:
        raise ValueError("GHPR baseline.csv must contain SHA and defect columns")
    if "SHA_FIXED" not in ghpr_df.columns or "SHA_BUG" not in ghpr_df.columns:
        raise ValueError("GHPR ghprdata.csv must contain SHA_FIXED and SHA_BUG columns")

    # GHPR baseline.csv encodes each commit pair as two rows with a 41-char
    # SHA: the first 40 chars are the real commit hash and the last character
    # is a suffix ('0' = fix commit, '1' = buggy commit). Both rows share the
    # same fix_sha, but some pairs carry conflicting labels (defect=0 for fix,
    # defect=1 for buggy). We drop the full conflicting group because a single
    # commit-text row cannot represent both labels without leaking pair identity.
    # For retained rows, the full 41-char SHA remains the unique module_id.
    baseline_df = baseline_df.copy()
    baseline_sha = _normalize_sha(baseline_df["SHA"])
    baseline_df["module_id"] = baseline_sha
    baseline_df["fix_sha"] = baseline_sha.str.slice(0, 40)
    baseline_df["label"] = baseline_df["defect"].astype(int)
    baseline_df["project_name"] = baseline_df.get("PROJECT_NAME", "GHPR")

    text_table = ghpr_df.copy()
    text_table["fix_sha"] = _normalize_sha(text_table["SHA_FIXED"])
    text_table["commit_text"] = text_table.apply(_compose_text, axis=1)
    # Preserve diff content separately so it can power a future diff-only
    # feature branch without contaminating the commit-message TF-IDF.
    text_table["diff_text"] = text_table.apply(_compose_diff_text, axis=1)
    text_table = text_table[["fix_sha", "commit_text", "diff_text"]].copy()
    text_table = text_table[text_table["fix_sha"].astype(str).str.strip().ne("")]
    text_table = text_table.drop_duplicates(subset=["fix_sha"], keep="first")

    merged = baseline_df.merge(text_table, on="fix_sha", how="left")
    merged["commit_text"] = merged["commit_text"].fillna("").astype(str)
    merged["diff_text"] = merged["diff_text"].fillna("").astype(str)
    merged["dataset_name"] = "ghpr"
    merged["ghpr_pair_policy"] = GHPR_PAIR_POLICY

    rows_before_policy = int(len(merged))
    conflicting_fix_sha_mask = merged.groupby("fix_sha")["label"].transform("nunique") > 1
    conflicting_fix_sha_count = int(merged.loc[conflicting_fix_sha_mask, "fix_sha"].nunique())
    conflicting_rows_dropped = int(conflicting_fix_sha_mask.sum())
    if conflicting_rows_dropped:
        merged = merged.loc[~conflicting_fix_sha_mask].copy()

    merged.attrs["ghpr_pair_policy"] = GHPR_PAIR_POLICY
    merged.attrs["ghpr_rows_before_pair_policy"] = rows_before_policy
    merged.attrs["ghpr_rows_after_pair_policy"] = int(len(merged))
    merged.attrs["ghpr_conflicting_fix_sha_count"] = conflicting_fix_sha_count
    merged.attrs["ghpr_conflicting_rows_dropped"] = conflicting_rows_dropped

    metric_columns = [col for col in GHPR_METRIC_COLUMNS if col in merged.columns]
    if not metric_columns:
        raise ValueError("GHPR dataset does not contain expected metric columns from baseline.csv")

    return merged

def validate_hybrid_source_frame(df: pd.DataFrame) -> None:
    """Validate minimum columns before hybrid feature fitting."""
    if df.empty:
        raise ValueError("The input DataFrame is empty after source-specific validation policies.")
    if "label" not in df.columns:
        raise ValueError("The input DataFrame must contain a 'label' column.")
    if "commit_text" not in df.columns:
        raise ValueError("The input DataFrame must contain a 'commit_text' column for TF-IDF training.")

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
        "num_metrics_features": int(metadata.get("metrics_num_features", metadata.get("num_features", 0))),
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
    logger.info("Default metrics (%s): %s", len(metrics), ", ".join(metrics))

def should_use_stratify(y: Any, use_stratify: bool) -> bool:
    class_counts = y.value_counts()
    return bool(use_stratify and y.nunique() > 1 and not class_counts.empty and class_counts.min() >= 2)

def _split_native_jitline_frames(
    df: pd.DataFrame,
    val_size: float,
    random_seed: int,
    use_stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Use the native train/(val/)test marker, carving validation off train when missing."""
    if "jitline_split" not in df.columns:
        raise ValueError("Dataset is missing the jitline_split column required for the native split.")

    normalized = df["jitline_split"].astype(str).str.strip().str.lower()
    train_pool = df.loc[normalized == "train"].copy()
    test_df = df.loc[normalized == "test"].copy()
    native_val_df = df.loc[normalized.isin({"val", "valid", "validation"})].copy()
    if train_pool.empty or test_df.empty:
        raise ValueError("Native split needs both train and test rows.")

    if not native_val_df.empty:
        return train_pool.copy(), native_val_df.copy(), test_df.copy(), "jitline_native_split"

    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1 for native split when no native val partition exists.")

    stratify_labels = train_pool["label"] if should_use_stratify(train_pool["label"], use_stratify) else None
    train_df, val_df = train_test_split(
        train_pool,
        test_size=val_size,
        random_state=random_seed,
        stratify=stratify_labels,
    )
    return train_df.copy(), val_df.copy(), test_df.copy(), "jitline_native_split"

def split_dataset_frames(
    df: pd.DataFrame,
    dataset_name: str,
    test_size: float,
    val_size: float,
    random_seed: int,
    use_stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Return train/val/test frames, preferring saved split artifacts."""
    if dataset_name in JITLINE_DATASETS and "jitline_split" in df.columns:
        return _split_native_jitline_frames(df, val_size=val_size, random_seed=random_seed, use_stratify=use_stratify)

    split_dir = SPLITS_DIR / dataset_name
    split_paths = [split_dir / "train_ids.csv", split_dir / "val_ids.csv", split_dir / "test_ids.csv"]
    if all(path.exists() for path in split_paths):
        train_df, val_df, test_df = reconstruct_split_frames(df, *split_paths)
        return train_df, val_df, test_df, "saved_split"

    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError("Fresh split requires 0 < test_size, val_size and test_size + val_size < 1")

    holdout_size = test_size + val_size
    stratify_labels = df["label"] if should_use_stratify(df["label"], use_stratify) else None
    train_df, holdout_df = train_test_split(df, test_size=holdout_size, random_state=random_seed, stratify=stratify_labels)
    holdout_stratify = holdout_df["label"] if should_use_stratify(holdout_df["label"], use_stratify) else None
    relative_test_size = test_size / holdout_size
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=relative_test_size,
        random_state=random_seed,
        stratify=holdout_stratify,
    )
    return train_df.copy(), val_df.copy(), test_df.copy(), "fresh_split"

def run_hybrid_tfidf_training() -> None:
    """Execute the full hybrid metrics + TF-IDF training flow."""
    ensure_project_dirs()
    HYBRID_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_training_config()
    random_seed = int(config.get("project", {}).get("random_seed", 42))
    default_metrics = list(config.get("features", {}).get("metrics", []))
    text_config = config.get("features", {}).get("text", {})
    tfidf_max_features = int(text_config.get("tfidf_max_features", 5000))
    tfidf_ngram_range = tuple(text_config.get("tfidf_ngram_range", [1, 2]))
    model_candidates = list(config.get("models", {}).get("candidates", ["rf"]))
    test_size = float(config.get("split", {}).get("test_size", 0.2))
    val_size = float(config.get("split", {}).get("val_size", 0.1))
    use_stratify = bool(config.get("split", {}).get("stratify", True))

    validate_training_configuration(default_metrics, model_candidates)
    set_global_seed(random_seed)
    log_training_configuration(default_metrics, model_candidates, random_seed, test_size, use_stratify)

    processed_files = discover_processed_datasets()
    logger.info("Selected %s dataset(s) for hybrid training.", len(processed_files))

    all_results: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    all_feature_coverage_records: list[dict[str, Any]] = []

    for dataset_path in processed_files:
        dataset_name = build_processed_dataset_name(dataset_path)
        metrics = resolve_metrics_for_dataset(config, dataset_name, default_metrics)
        if not metrics:
            all_failures.append({
                "dataset_name": dataset_name,
                "stage": "metric_resolution",
                "error": "No metric columns configured for this dataset.",
                "source_file": str(dataset_path),
            })
            continue

        source_policy_metadata: dict[str, Any] = {}
        try:
            if dataset_path == GHPR_PROCESSED_PATH:
                df = prepare_ghpr_hybrid_frame()
                source_policy_metadata = _ghpr_pair_metadata(df)
                feature_source = "ghpr_raw"
            else:
                df = read_parquet(dataset_path)
                feature_source = str(dataset_path)

            validate_hybrid_source_frame(df)
            train_df, val_df, test_df, split_mode = split_dataset_frames(
                df=df,
                dataset_name=dataset_name,
                test_size=test_size,
                val_size=val_size,
                random_seed=random_seed,
                use_stratify=use_stratify,
            )
            feature_spec = fit_hybrid_tfidf_feature_spec(
                train_df,
                metrics,
                text_column="commit_text",
                max_features=tfidf_max_features,
                ngram_range=(int(tfidf_ngram_range[0]), int(tfidf_ngram_range[1])),
            )
            X_train, y_train, feature_metadata = build_hybrid_tfidf_training_frame(train_df, feature_spec)
            X_val, y_val, _ = build_hybrid_tfidf_training_frame(val_df, feature_spec)
            X_test, y_test, _ = build_hybrid_tfidf_training_frame(test_df, feature_spec)
            coverage_record = {
                **build_feature_coverage_record(dataset_name, feature_metadata),
                "configured_metrics": ",".join(metrics),
                "configured_models": ",".join(model_candidates),
                "random_seed": int(random_seed),
                "test_size": float(test_size),
                "val_size": float(val_size),
                "stratify_enabled": bool(use_stratify),
                "split_mode": split_mode,
                "feature_mode": "metrics+tfidf",
                "source_file": feature_source,
                **source_policy_metadata,
            }
            all_feature_coverage_records.append(coverage_record)

            if X_train.empty or X_train.shape[1] == 0:
                all_failures.append({
                    "dataset_name": dataset_name,
                    "stage": "tfidf_feature_building",
                    "error": "No usable commit_text for TF-IDF features or no hybrid features remained after preprocessing.",
                    "source_file": feature_source,
                    **source_policy_metadata,
                })
                continue
            if not bool(feature_metadata.get("uses_commit_text", False)) or int(feature_metadata.get("tfidf_num_features", 0)) == 0:
                all_failures.append({
                    "dataset_name": dataset_name,
                    "stage": "tfidf_feature_building",
                    "error": "Hybrid TF-IDF training requires at least one fitted commit-text feature; skipping metrics-only fallback.",
                    "source_file": feature_source,
                    "split_mode": split_mode,
                    **source_policy_metadata,
                })
                continue
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                all_failures.append({
                    "dataset_name": dataset_name,
                    "stage": "dataset_validation",
                    "error": "Train and test labels must contain at least two classes for training.",
                    "source_file": feature_source,
                    **source_policy_metadata,
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
                        X_val=X_val,
                        y_val=y_val,
                        feature_preprocessor=feature_spec,
                        threshold_strategy=resolve_threshold_strategy(config, dataset_name),
                    )
                    model_path = HYBRID_MODELS_DIR / f"{model_name}_{dataset_name}.joblib"
                    save_model(model, model_path)
                    result.update(
                        {
                            "source_file": feature_source,
                            "random_seed": int(random_seed),
                            "test_size": float(test_size),
                            "val_size": float(val_size),
                            "stratified_split": bool(use_stratify),
                            "split_mode": split_mode,
                            "model_path": str(model_path),
                            "feature_mode": "metrics+tfidf",
                            "feature_family": result.get("feature_family", feature_metadata.get("feature_family", "metrics_only")),
                            "feature_set": result.get("feature_set", feature_metadata.get("feature_set", feature_metadata.get("feature_family", "metrics_only"))),
                            "uses_commit_text": bool(feature_metadata.get("uses_commit_text", False)),
                            "artifact_stage": "training",
                            "artifact_schema_version": "paper-v1",
                            "artifact_group_key": f"{dataset_name}::{model_name}",
                            "artifact_id": f"{dataset_name}::{model_name}::training",
                            "source_results_table": str(RESULTS_TABLE_PATH),
                            "num_val_rows": int(len(val_df)),
                            "configured_metrics": ",".join(metrics),
                            **source_policy_metadata,
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
                        **source_policy_metadata,
                    })
        except Exception as exc:
            all_failures.append({
                "dataset_name": dataset_name,
                "stage": "dataset_loading",
                "error": str(exc),
                "source_file": str(dataset_path),
                **source_policy_metadata,
            })

    coverage_df = build_results_table(all_feature_coverage_records)
    results_df = build_results_table(all_results)
    if results_df.empty and len(results_df.columns) == 0:
        results_df = pd.DataFrame(columns=HYBRID_RESULTS_COLUMNS)

    write_csv(coverage_df, FEATURE_COVERAGE_PATH)
    write_csv(results_df, RESULTS_TABLE_PATH)
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
