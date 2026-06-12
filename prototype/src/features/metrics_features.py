"""Feature engineering for software metrics and commit text."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.features.commit_tfidf import build_tfidf_features, normalize_commit_text


DEFAULT_METRIC_REGISTRY = {
    "core": ["loc", "v(g)", "ev(g)", "iv(g)", "branchCount"],
    "paper_extended": ["loc", "v(g)", "ev(g)", "iv(g)", "branchCount", "coupling", "cohesion", "code_churn"],
}

COMMIT_TEXT_COLUMNS = ["commit_text", "commit_message", "commit_msg", "message", "log", "commit"]


def get_available_metrics(df: pd.DataFrame, metrics: list[str]) -> tuple[list[str], list[str]]:
    """Return metric columns that exist in the dataset and those that are missing."""
    available = [col for col in metrics if col in df.columns]
    missing = [col for col in metrics if col not in df.columns]
    return available, missing


def summarize_metric_coverage(df: pd.DataFrame, metrics: list[str]) -> dict[str, Any]:
    """Summarize the availability of configured metrics in the dataset."""
    available, missing = get_available_metrics(df, metrics)
    coverage_ratio = len(available) / len(metrics) if metrics else None
    return {
        "configured_metrics": list(metrics),
        "available_metrics": available,
        "missing_metrics": missing,
        "coverage_ratio": coverage_ratio,
    }


def build_metrics_features(
    df: pd.DataFrame,
    metrics: list[str],
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Build a numeric metrics-only feature matrix for baseline models."""
    available, missing = get_available_metrics(df, metrics)

    if not available:
        empty = pd.DataFrame(index=df.index)
        metadata = {
            "selected_metrics": [],
            "missing_metrics": missing,
            "dropped_all_nan_metrics": [],
            "num_features": 0,
            "metric_group": "metrics_only",
        }
        return (empty, metadata) if return_metadata else empty

    feature_df = df[available].copy()
    dropped_all_nan_metrics: list[str] = []

    for col in list(feature_df.columns):
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
        if feature_df[col].isna().all():
            dropped_all_nan_metrics.append(col)
            feature_df = feature_df.drop(columns=[col])
            continue
        if feature_df[col].isna().any():
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())

    metadata = {
        "selected_metrics": list(feature_df.columns),
        "missing_metrics": missing,
        "dropped_all_nan_metrics": dropped_all_nan_metrics,
        "num_features": feature_df.shape[1],
        "metric_group": "metrics_only",
    }
    return (feature_df, metadata) if return_metadata else feature_df


def build_commit_text_features(
    df: pd.DataFrame,
    return_metadata: bool = False,
    max_features: int = 500,
    ngram_range: tuple[int, int] = (1, 2),
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Build a TF-IDF feature block from any available commit text column."""
    text_col = next((column for column in COMMIT_TEXT_COLUMNS if column in df.columns), None)
    if text_col is None:
        empty = pd.DataFrame(index=df.index)
        metadata = {"text_column": None, "num_features": 0, "feature_group": "commit_text", "used_fallback": True}
        return (empty, metadata) if return_metadata else empty

    text_series = normalize_commit_text(df[text_col])
    vectorizer, tfidf_df = build_tfidf_features(text_series, max_features=max_features, ngram_range=ngram_range)
    metadata = {
        "text_column": text_col,
        "num_features": int(tfidf_df.shape[1]),
        "feature_group": "commit_text",
        "used_fallback": tfidf_df.empty,
        "vocabulary_size": int(len(getattr(vectorizer, "vocabulary_", {}) or {})),
    }
    return (tfidf_df, metadata) if return_metadata else tfidf_df


def build_metrics_training_frame(
    df: pd.DataFrame,
    metrics: list[str],
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Return X, y, and feature metadata for metrics-only training.

    This helper expects a cleaned dataset that already contains a `label` column.
    """
    if "label" not in df.columns:
        raise ValueError("The input DataFrame must contain a 'label' column.")

    X, metadata = build_metrics_features(df, metrics, return_metadata=True)
    y = pd.to_numeric(df["label"], errors="coerce")

    if y.isna().any():
        raise ValueError("The 'label' column contains non-numeric values after preprocessing.")

    metadata["num_rows"] = len(df)
    metadata["label_distribution"] = y.value_counts().to_dict()
    return X, y.astype(int), metadata


def build_hybrid_training_frame(
    df: pd.DataFrame,
    metrics: list[str],
    max_commit_features: int = 500,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Return a combined metrics + commit-text training frame.

    This is the paper-facing helper for the combined feature family.
    """
    metrics_df, metrics_meta = build_metrics_features(df, metrics, return_metadata=True)
    commit_df, commit_meta = build_commit_text_features(df, return_metadata=True, max_features=max_commit_features)

    if "label" not in df.columns:
        raise ValueError("The input DataFrame must contain a 'label' column.")

    y = pd.to_numeric(df["label"], errors="coerce")
    if y.isna().any():
        raise ValueError("The 'label' column contains non-numeric values after preprocessing.")

    feature_df = pd.concat([metrics_df, commit_df], axis=1)
    feature_df = feature_df.loc[:, ~feature_df.columns.duplicated()]
    if not feature_df.empty:
        feature_df = feature_df.fillna(0.0)

    metadata = {
        "num_rows": len(df),
        "label_distribution": y.value_counts().to_dict(),
        "metrics_metadata": metrics_meta,
        "commit_metadata": commit_meta,
        "num_features": int(feature_df.shape[1]),
        "feature_family": "metrics_plus_commit_text",
    }
    return feature_df, y.astype(int), metadata


def get_default_metric_registry() -> dict[str, list[str]]:
    """Return the canonical metric groups used by the project."""
    return {key: list(values) for key, values in DEFAULT_METRIC_REGISTRY.items()}
