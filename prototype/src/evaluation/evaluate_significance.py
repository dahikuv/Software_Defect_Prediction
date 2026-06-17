"""Bootstrap CIs and paired significance tests for the final evaluation report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.split import reconstruct_split_frames
from src.evaluation.significance import (
    DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAP_ITERS,
    DEFAULT_PERMUTATION_ITERS,
    benjamini_hochberg,
    bootstrap_ci_metric,
    bootstrap_paired_delta,
    delong_test_auc,
    paired_permutation_test,
)
from src.evaluation.compare import rank_models_by_dataset
from src.models.predict import load_model
from src.utils.io import read_csv, read_parquet
from src.utils.logging import get_logger
from src.utils.paths import PROCESSED_DATA_DIR, RESULTS_TABLES_DIR, SPLITS_DIR
from src.utils.provenance import artifact_uses_commit_text

logger = get_logger(__name__)

PROMISE_DATASETS = {"cm1", "jm1", "kc1", "pc1"}
JITLINE_DATASETS = {"openstack", "qt", "jitfine"}
HYBRID_RESULTS_PATH = RESULTS_TABLES_DIR / "hybrid_tfidf_results.csv"
FINAL_SELECTION_PATH = RESULTS_TABLES_DIR / "final_models_by_dataset.csv"
BEST_MODELS_PATH = RESULTS_TABLES_DIR / "best_models_by_dataset.csv"
TUNED_BEST_PATH = RESULTS_TABLES_DIR / "metrics_tuned_best.csv"
SIGNIFICANCE_OUTPUT_PATH = RESULTS_TABLES_DIR / "evaluation_significance.csv"

METRICS_TO_REPORT = ("auc", "f1", "precision", "recall")


def _load_test_frame(dataset_name: str, source_file: str) -> pd.DataFrame | None:
    name = dataset_name.lower()
    source_path = Path(source_file) if source_file else None
    if name in PROMISE_DATASETS:
        clean_path = PROCESSED_DATA_DIR / f"{name}_clean.parquet"
        if not clean_path.exists():
            logger.warning("PROMISE dataset %s missing cleaned parquet", name)
            return None
        cleaned = read_parquet(clean_path)
        split_dir = SPLITS_DIR / name
        try:
            _, _, test_df = reconstruct_split_frames(
                cleaned,
                split_dir / "train_ids.csv",
                split_dir / "val_ids.csv",
                split_dir / "test_ids.csv",
                id_col="module_id",
            )
        except Exception as exc:
            logger.warning("Cannot reconstruct PROMISE split for %s: %s", name, exc)
            return None
        return test_df
    if name in JITLINE_DATASETS:
        clean_path = source_path if source_path and source_path.exists() else PROCESSED_DATA_DIR / f"{name}_clean.parquet"
        if not clean_path.exists():
            logger.warning("JITLine dataset %s missing cleaned parquet", name)
            return None
        df = read_parquet(clean_path)
        if "jitline_split" not in df.columns:
            logger.warning("JITLine dataset %s missing native split column", name)
            return None
        normalized = df["jitline_split"].astype(str).str.strip().str.lower()
        return df.loc[normalized == "test"].copy()
    logger.warning("Unsupported dataset for significance: %s", name)
    return None


def _resolve_threshold(value: Any, fallback: float = 0.5) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(threshold):
        return fallback
    return threshold


def _predict_proba(bundle: Any, frame: pd.DataFrame) -> np.ndarray | None:
    try:
        proba = bundle.predict_proba(frame)
    except Exception as exc:
        logger.warning("predict_proba failed: %s", exc)
        return None
    arr = np.asarray(proba)
    if arr.ndim != 2 or arr.shape[0] != len(frame):
        return None
    classes = list(getattr(bundle.estimator, "classes_", []))
    if len(classes) == arr.shape[1] and 1 in classes:
        positive_index = classes.index(1)
    else:
        positive_index = arr.shape[1] - 1
    return arr[:, positive_index]


def _best_ranked_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    ranked = rank_models_by_dataset(df)
    if ranked.empty:
        return None
    return ranked.iloc[0]


def _select_metrics_reference_row(dataset_name: str, tuned_df: pd.DataFrame, best_df: pd.DataFrame) -> pd.Series | None:
    """Choose the metrics-only reference for a selected hybrid final model."""
    for source_df in [tuned_df, best_df]:
        if source_df.empty or "dataset_name" not in source_df.columns:
            continue
        subset = source_df[source_df["dataset_name"].astype(str).str.lower() == dataset_name].copy()
        if subset.empty:
            continue
        if "uses_commit_text" in subset.columns:
            subset = subset[~subset.apply(lambda row: artifact_uses_commit_text(row.to_dict()), axis=1)]
        row = _best_ranked_row(subset)
        if row is not None:
            return row
    return None


def _select_hybrid_comparison_row(dataset_name: str, selected_row: pd.Series, hybrid_df: pd.DataFrame) -> pd.Series | None:
    if artifact_uses_commit_text(selected_row.to_dict()):
        return selected_row
    if hybrid_df.empty or "dataset_name" not in hybrid_df.columns:
        return None
    subset = hybrid_df[hybrid_df["dataset_name"].astype(str).str.lower() == dataset_name].copy()
    if subset.empty:
        return None
    subset = subset[subset.apply(lambda row: artifact_uses_commit_text(row.to_dict()), axis=1)]
    return _best_ranked_row(subset)


def _format_record(
    *,
    dataset_name: str,
    comparison: str,
    metric: str,
    baseline_summary: dict[str, Any],
    hybrid_summary: dict[str, Any] | None,
    delta_summary: dict[str, Any] | None,
    permutation_summary: dict[str, Any] | None,
    delong_summary: dict[str, Any] | None,
    n_test_rows: int,
    n_test_positive: int,
    n_bootstrap_iters: int,
    n_permutation_iters: int,
    alpha: float,
    baseline_label: str,
    hybrid_label: str | None,
) -> dict[str, Any]:
    nan = float("nan")
    record: dict[str, Any] = {
        "dataset_name": dataset_name,
        "comparison": comparison,
        "metric": metric,
        "n_test_rows": int(n_test_rows),
        "n_test_positive": int(n_test_positive),
        "n_bootstrap_iters": int(n_bootstrap_iters),
        "n_permutation_iters": int(n_permutation_iters),
        "alpha": float(alpha),
        "baseline_model": baseline_label,
        "hybrid_model": hybrid_label or "",
        "point_baseline": float(baseline_summary.get("point", nan)),
        "ci_baseline_low": float(baseline_summary.get("ci_low", nan)),
        "ci_baseline_high": float(baseline_summary.get("ci_high", nan)),
        "point_hybrid": float((hybrid_summary or {}).get("point", nan)),
        "ci_hybrid_low": float((hybrid_summary or {}).get("ci_low", nan)),
        "ci_hybrid_high": float((hybrid_summary or {}).get("ci_high", nan)),
        "delta": nan,
        "delta_ci_low": nan,
        "delta_ci_high": nan,
        "permutation_p_value": nan,
        "permutation_n_iter_effective": 0,
        "delong_z": nan,
        "delong_p_value": nan,
        "delong_auc_diff": nan,
    }
    if delta_summary is not None:
        record["delta"] = float(delta_summary.get("delta", nan))
        record["delta_ci_low"] = float(delta_summary.get("delta_ci_low", nan))
        record["delta_ci_high"] = float(delta_summary.get("delta_ci_high", nan))
    if permutation_summary is not None:
        record["permutation_p_value"] = float(permutation_summary.get("p_value", nan))
        record["permutation_n_iter_effective"] = int(permutation_summary.get("n_iter_effective", 0))
    if delong_summary is not None and metric == "auc":
        record["delong_z"] = float(delong_summary.get("z", nan))
        record["delong_p_value"] = float(delong_summary.get("p_value", nan))
        record["delong_auc_diff"] = float(delong_summary.get("auc_diff", nan))
    return record


def build_significance_table(
    n_bootstrap_iters: int = DEFAULT_BOOTSTRAP_ITERS,
    n_permutation_iters: int = DEFAULT_PERMUTATION_ITERS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute bootstrap CIs and paired significance tests for the final selection."""
    if not FINAL_SELECTION_PATH.exists():
        logger.warning("Final selection table missing: %s", FINAL_SELECTION_PATH)
        return pd.DataFrame()

    final_df = read_csv(FINAL_SELECTION_PATH)
    if final_df.empty:
        return pd.DataFrame()

    hybrid_df = read_csv(HYBRID_RESULTS_PATH) if HYBRID_RESULTS_PATH.exists() else pd.DataFrame()
    tuned_df = read_csv(TUNED_BEST_PATH) if TUNED_BEST_PATH.exists() else pd.DataFrame()
    best_df = read_csv(BEST_MODELS_PATH) if BEST_MODELS_PATH.exists() else pd.DataFrame()

    records: list[dict[str, Any]] = []
    for _, row in final_df.iterrows():
        dataset_name = str(row.get("dataset_name", "")).strip().lower()
        if not dataset_name:
            continue
        reference_row = row
        selected_uses_commit_text = artifact_uses_commit_text(row.to_dict())
        if selected_uses_commit_text:
            metrics_reference = _select_metrics_reference_row(dataset_name, tuned_df, best_df)
            if metrics_reference is None:
                logger.warning("Skipping %s: selected hybrid has no metrics-only reference row", dataset_name)
                continue
            reference_row = metrics_reference

        baseline_path = str(reference_row.get("model_path", ""))
        if not baseline_path or not Path(baseline_path).exists():
            logger.warning("Baseline bundle missing for %s: %s", dataset_name, baseline_path)
            continue
        baseline_threshold = _resolve_threshold(reference_row.get("decision_threshold"))
        try:
            baseline_bundle = load_model(baseline_path)
        except Exception as exc:
            logger.warning("Cannot load baseline bundle for %s: %s", dataset_name, exc)
            continue

        test_source = str(row.get("source_file", "") or reference_row.get("source_file", ""))
        test_df = _load_test_frame(dataset_name, test_source)
        if test_df is None or test_df.empty or "label" not in test_df.columns:
            logger.warning("Skipping %s: no usable test frame", dataset_name)
            continue
        y_true = test_df["label"].astype(int).to_numpy()
        baseline_scores = _predict_proba(baseline_bundle, test_df)
        if baseline_scores is None:
            continue

        baseline_label = "_".join(
            part for part in [str(reference_row.get("model", "")), str(reference_row.get("training_mode", ""))] if part
        )
        n_test_rows = int(len(test_df))
        n_pos = int(y_true.sum())

        baseline_summaries: dict[str, dict[str, Any]] = {}
        for metric in METRICS_TO_REPORT:
            baseline_summaries[metric] = bootstrap_ci_metric(
                y_true,
                baseline_scores,
                threshold=baseline_threshold,
                metric_name=metric,
                n_iter=n_bootstrap_iters,
                seed=seed,
                alpha=alpha,
            )
            records.append(
                _format_record(
                    dataset_name=dataset_name,
                    comparison="baseline_only",
                    metric=metric,
                    baseline_summary=baseline_summaries[metric],
                    hybrid_summary=None,
                    delta_summary=None,
                    permutation_summary=None,
                    delong_summary=None,
                    n_test_rows=n_test_rows,
                    n_test_positive=n_pos,
                    n_bootstrap_iters=n_bootstrap_iters,
                    n_permutation_iters=n_permutation_iters,
                    alpha=alpha,
                    baseline_label=baseline_label,
                    hybrid_label=None,
                )
            )

        if dataset_name not in JITLINE_DATASETS:
            continue

        hybrid_row = _select_hybrid_comparison_row(dataset_name, row, hybrid_df)
        if hybrid_row is None:
            continue
        hybrid_path = str(hybrid_row.get("model_path", ""))
        if not hybrid_path or not Path(hybrid_path).exists():
            logger.warning("Hybrid bundle missing for %s: %s", dataset_name, hybrid_path)
            continue
        try:
            hybrid_bundle = load_model(hybrid_path)
        except Exception as exc:
            logger.warning("Cannot load hybrid bundle for %s: %s", dataset_name, exc)
            continue
        hybrid_threshold = _resolve_threshold(hybrid_row.get("decision_threshold"))
        hybrid_scores = _predict_proba(hybrid_bundle, test_df)
        if hybrid_scores is None:
            continue

        hybrid_label = f"{hybrid_row.get('model', '')}_hybrid_tfidf".strip("_")
        for metric in METRICS_TO_REPORT:
            hybrid_summary = bootstrap_ci_metric(
                y_true,
                hybrid_scores,
                threshold=hybrid_threshold,
                metric_name=metric,
                n_iter=n_bootstrap_iters,
                seed=seed,
                alpha=alpha,
            )
            delta_summary = bootstrap_paired_delta(
                y_true,
                baseline_scores,
                hybrid_scores,
                threshold_a=baseline_threshold,
                threshold_b=hybrid_threshold,
                metric_name=metric,
                n_iter=n_bootstrap_iters,
                seed=seed,
                alpha=alpha,
            )
            permutation_summary = paired_permutation_test(
                y_true,
                baseline_scores,
                hybrid_scores,
                threshold_a=baseline_threshold,
                threshold_b=hybrid_threshold,
                metric_name=metric,
                n_iter=n_permutation_iters,
                seed=seed,
            )
            delong_summary = delong_test_auc(y_true, baseline_scores, hybrid_scores) if metric == "auc" else None
            records.append(
                _format_record(
                    dataset_name=dataset_name,
                    comparison="baseline_vs_hybrid_tfidf",
                    metric=metric,
                    baseline_summary=baseline_summaries[metric],
                    hybrid_summary=hybrid_summary,
                    delta_summary=delta_summary,
                    permutation_summary=permutation_summary,
                    delong_summary=delong_summary,
                    n_test_rows=n_test_rows,
                    n_test_positive=n_pos,
                    n_bootstrap_iters=n_bootstrap_iters,
                    n_permutation_iters=n_permutation_iters,
                    alpha=alpha,
                    baseline_label=baseline_label,
                    hybrid_label=hybrid_label,
                )
            )

    if not records:
        return pd.DataFrame()
    table = pd.DataFrame(records)

    # Multiple-comparison correction across the family of paired hypothesis
    # tests (4 metrics x N datasets). Baseline-only rows carry no p-value, so
    # they stay NaN and never consume a comparison slot.
    comparison_mask = table["comparison"] == "baseline_vs_hybrid_tfidf"

    perm_adjusted = pd.Series(float("nan"), index=table.index)
    perm_rejected = pd.Series(False, index=table.index)
    if comparison_mask.any():
        perm_result = benjamini_hochberg(
            table.loc[comparison_mask, "permutation_p_value"].tolist(),
            alpha=alpha,
        )
        perm_adjusted.loc[comparison_mask] = perm_result["p_adjusted"]
        perm_rejected.loc[comparison_mask] = perm_result["rejected"]
    table["permutation_p_value_bh"] = perm_adjusted
    table["permutation_significant_bh"] = perm_rejected

    delong_mask = comparison_mask & (table["metric"] == "auc")
    delong_adjusted = pd.Series(float("nan"), index=table.index)
    delong_rejected = pd.Series(False, index=table.index)
    if delong_mask.any():
        delong_result = benjamini_hochberg(
            table.loc[delong_mask, "delong_p_value"].tolist(),
            alpha=alpha,
        )
        delong_adjusted.loc[delong_mask] = delong_result["p_adjusted"]
        delong_rejected.loc[delong_mask] = delong_result["rejected"]
    table["delong_p_value_bh"] = delong_adjusted
    table["delong_significant_bh"] = delong_rejected

    return table


def write_significance_table(
    n_bootstrap_iters: int = DEFAULT_BOOTSTRAP_ITERS,
    n_permutation_iters: int = DEFAULT_PERMUTATION_ITERS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> Path | None:
    """Build the significance table and persist it under results/tables."""
    table = build_significance_table(
        n_bootstrap_iters=n_bootstrap_iters,
        n_permutation_iters=n_permutation_iters,
        alpha=alpha,
        seed=seed,
    )
    if table.empty:
        logger.warning("Significance table is empty; nothing was written.")
        return None
    SIGNIFICANCE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(SIGNIFICANCE_OUTPUT_PATH, index=False, encoding="utf-8")
    logger.info("Saved evaluation significance table to %s", SIGNIFICANCE_OUTPUT_PATH)
    return SIGNIFICANCE_OUTPUT_PATH
