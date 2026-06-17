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

def _confidence_series(df: pd.DataFrame) -> pd.Series:
    if "probability" not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    return df["probability"].where(df["prediction"] == 1, 1 - df["probability"])

def select_representative_cases(
    df: pd.DataFrame,
    top_k: int = 5,
    per_error_type: bool = True,
) -> pd.DataFrame:
    """Return representative cases for each error type when possible.

    When per_error_type is True, the result contains up to top_k rows for each
    of tp, tn, fp, fn. The legacy correct/incorrect grouping is also kept for
    backward compatibility with existing tables.
    """
    result = df.copy()
    result["confidence"] = _confidence_series(result)
    sort_columns = ["confidence"] if "probability" in result.columns else []

    frames: list[pd.DataFrame] = []
    if per_error_type and "error_type" in result.columns:
        for error_type in ["tp", "fn", "fp", "tn"]:
            subset = result[result["error_type"] == error_type].copy()
            if subset.empty:
                continue
            if sort_columns:
                subset = subset.sort_values(sort_columns, ascending=False, na_position="last")
            subset = subset.head(top_k).copy()
            subset["representative_group"] = error_type
            frames.append(subset)

    correct = result[result["is_correct"] == True].copy()  # noqa: E712
    incorrect = result[result["is_correct"] == False].copy()  # noqa: E712
    if sort_columns:
        correct = correct.sort_values(sort_columns, ascending=False, na_position="last")
        incorrect = incorrect.sort_values(sort_columns, ascending=False, na_position="last")
    correct = correct.head(top_k).copy()
    incorrect = incorrect.head(top_k).copy()
    correct["representative_group"] = "correct"
    incorrect["representative_group"] = "incorrect"
    frames.extend([correct, incorrect])

    return pd.concat(frames, ignore_index=True, sort=False)