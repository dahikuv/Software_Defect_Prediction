"""Helpers for selecting representative correct and incorrect predictions."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_error_analysis_frame(df: pd.DataFrame, label_col: str = "label", pred_col: str = "prediction") -> pd.DataFrame:
    """Annotate each row with prediction outcome labels."""
    analysis = df.copy()
    analysis["is_correct"] = analysis[label_col] == analysis[pred_col]
    analysis["error_type"] = "correct"

    false_positive = (analysis[label_col] == 0) & (analysis[pred_col] == 1)
    false_negative = (analysis[label_col] == 1) & (analysis[pred_col] == 0)
    true_positive = (analysis[label_col] == 1) & (analysis[pred_col] == 1)
    true_negative = (analysis[label_col] == 0) & (analysis[pred_col] == 0)

    analysis.loc[true_positive, "error_type"] = "tp"
    analysis.loc[true_negative, "error_type"] = "tn"
    analysis.loc[false_positive, "error_type"] = "fp"
    analysis.loc[false_negative, "error_type"] = "fn"
    return analysis


def build_error_summary(df: pd.DataFrame, dataset_name: str, model_name: str) -> dict[str, Any]:
    """Summarize correct and incorrect predictions for one dataset/model pair."""
    summary = {
        "dataset_name": dataset_name,
        "model": model_name,
        "num_rows": int(len(df)),
        "num_correct": int(df["is_correct"].sum()),
        "num_incorrect": int((~df["is_correct"]).sum()),
        "tp": int((df["error_type"] == "tp").sum()),
        "tn": int((df["error_type"] == "tn").sum()),
        "fp": int((df["error_type"] == "fp").sum()),
        "fn": int((df["error_type"] == "fn").sum()),
    }
    if "probability" in df.columns:
        summary["mean_probability"] = float(df["probability"].mean())
    return summary


def select_representative_cases(df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Return representative high-confidence cases for discussion."""
    result = df.copy()
    if "probability" in result.columns:
        result["confidence"] = result["probability"].where(result["prediction"] == 1, 1 - result["probability"])
        result = result.sort_values(["is_correct", "confidence"], ascending=[True, False])
    return result.head(top_k * 2)

