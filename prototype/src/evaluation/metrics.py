"""Classification metric helpers."""

from __future__ import annotations

from typing import Iterable

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Iterable[float] | None = None,
) -> dict[str, float]:
    """Compute the main classification metrics used in the paper."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_score is None:
        metrics["auc"] = float("nan")
        return metrics

    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics
