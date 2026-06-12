"""Compare metrics-only and hybrid model outputs."""

from __future__ import annotations

import pandas as pd

CANONICAL_COMPARISON_COLUMNS = [
    "dataset_name",
    "model",
    "feature_family",
    "feature_set",
    "text_feature_column",
    "commit_text_column",
    "commit_text_available",
    "uses_commit_text",
    "artifact_schema_version",
    "artifact_stage",
    "artifact_group_key",
    "artifact_id",
    "source_results_table",
]


def compare_commit_impact(
    metrics_only: pd.DataFrame,
    hybrid: pd.DataFrame,
    on: str | list[str] = "model",
) -> pd.DataFrame:
    """Merge and compute deltas between metrics-only and commit-augmented runs."""
    join_keys = [on] if isinstance(on, str) else list(on)
    merged = metrics_only.merge(hybrid, on=join_keys, suffixes=("_metrics", "_hybrid"))

    merged["baseline_feature_family"] = merged.get("feature_family_metrics", "metrics_only")
    merged["hybrid_feature_family"] = merged.get("feature_family_hybrid", "metrics_plus_commit_text")

    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        metrics_col = f"{metric}_metrics"
        hybrid_col = f"{metric}_hybrid"
        if metrics_col in merged.columns and hybrid_col in merged.columns:
            merged[f"delta_{metric}"] = merged[hybrid_col] - merged[metrics_col]

    if "text_feature_column_hybrid" in merged.columns:
        merged["commit_text_source"] = merged["text_feature_column_hybrid"]
    return merged
