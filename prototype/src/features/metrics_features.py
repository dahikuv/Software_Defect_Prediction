"""Feature engineering for software metrics and commit text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

DEFAULT_METRIC_REGISTRY = {
    "core": ["loc", "v(g)", "ev(g)", "iv(g)", "branchCount"],
    "paper_extended": ["loc", "v(g)", "ev(g)", "iv(g)", "branchCount", "coupling", "cohesion", "code_churn"],
}

@dataclass
class MetricsFeatureSpec:
    """Train-fitted metric feature schema and imputation values."""

    configured_metrics: list[str]
    selected_metrics: list[str]
    missing_metrics: list[str]
    dropped_all_nan_metrics: list[str]
    medians: dict[str, float]
    metric_group: str = "metrics_only"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "configured_metrics": list(self.configured_metrics),
            "selected_metrics": list(self.selected_metrics),
            "missing_metrics": list(self.missing_metrics),
            "dropped_all_nan_metrics": list(self.dropped_all_nan_metrics),
            "num_features": len(self.selected_metrics),
            "metric_group": self.metric_group,
            "feature_family": "metrics_only",
            "feature_set": "metrics_only",
        }

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
    """Build a numeric metrics-only feature matrix for baseline models.

    WARNING: This fits the imputation spec (median values) and transforms the
    SAME frame. For modeling, fit on the TRAINING split only via
    ``fit_metrics_feature_spec`` and apply ``transform_metrics_features`` to
    val/test; fitting on a full, un-split dataset leaks test-row medians into
    training. Use this convenience helper for exploration/EDA only.
    """
    spec = fit_metrics_feature_spec(df, metrics)
    feature_df = transform_metrics_features(df, spec)
    metadata = spec.to_metadata()
    return (feature_df, metadata) if return_metadata else feature_df

def fit_metrics_feature_spec(df: pd.DataFrame, metrics: list[str]) -> MetricsFeatureSpec:
    """Fit a metric feature schema and median imputers on one training frame."""
    available, missing = get_available_metrics(df, metrics)

    if not available:
        return MetricsFeatureSpec(
            configured_metrics=list(metrics),
            selected_metrics=[],
            missing_metrics=missing,
            dropped_all_nan_metrics=[],
            medians={},
        )

    feature_df = df[available].copy()
    dropped_all_nan_metrics: list[str] = []
    selected_metrics: list[str] = []
    medians: dict[str, float] = {}

    for col in list(feature_df.columns):
        series = feature_df[col]
        if pd.api.types.is_bool_dtype(series):
            series = series.astype("float64")
        feature_df[col] = pd.to_numeric(series, errors="coerce").astype("float64")
        if feature_df[col].isna().all():
            dropped_all_nan_metrics.append(col)
            continue
        median = feature_df[col].median()
        medians[col] = 0.0 if pd.isna(median) else float(median)
        selected_metrics.append(col)

    return MetricsFeatureSpec(
        configured_metrics=list(metrics),
        selected_metrics=selected_metrics,
        missing_metrics=missing,
        dropped_all_nan_metrics=dropped_all_nan_metrics,
        medians=medians,
    )

def transform_metrics_features(df: pd.DataFrame, spec: MetricsFeatureSpec) -> pd.DataFrame:
    """Transform a frame using a train-fitted metrics feature spec."""
    feature_df = pd.DataFrame(index=df.index)
    for col in spec.selected_metrics:
        if col in df.columns:
            raw_series = df[col]
            if pd.api.types.is_bool_dtype(raw_series):
                raw_series = raw_series.astype("float64")
            series = pd.to_numeric(raw_series, errors="coerce").astype("float64")
        else:
            series = pd.Series(index=df.index, dtype="float64")
        feature_df[col] = series.fillna(float(spec.medians.get(col, 0.0))).astype("float64")
    return feature_df

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

def transform_metrics_training_frame(
    df: pd.DataFrame,
    spec: MetricsFeatureSpec,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Return X, y, and metadata using a train-fitted metrics feature spec."""
    if "label" not in df.columns:
        raise ValueError("The input DataFrame must contain a 'label' column.")

    X = transform_metrics_features(df, spec)
    y = pd.to_numeric(df["label"], errors="coerce")
    if y.isna().any():
        raise ValueError("The 'label' column contains non-numeric values after preprocessing.")

    metadata = spec.to_metadata()
    metadata["num_rows"] = len(df)
    metadata["label_distribution"] = y.value_counts().to_dict()
    return X, y.astype(int), metadata

def get_default_metric_registry() -> dict[str, list[str]]:
    """Return the canonical metric groups used by the project."""
    return {key: list(values) for key, values in DEFAULT_METRIC_REGISTRY.items()}
