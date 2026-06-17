"""Classification metric helpers and decision-threshold selectors."""

from __future__ import annotations

from typing import Any, Iterable

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

THRESHOLD_STRATEGIES = (
    "recall_with_precision_floor",
    "f1_optimal",
    "youden_j",
)
DEFAULT_THRESHOLD_STRATEGY = "recall_with_precision_floor"


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


def compute_metrics_at_threshold(
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float,
) -> dict[str, float]:
    """Compute binary metrics after converting probabilities with a threshold."""
    scores = list(y_score)
    y_pred = [1 if float(score) >= float(threshold) else 0 for score in scores]
    return compute_classification_metrics(y_true, y_pred, scores)


def _empty_threshold_summary(precision_floor: float, strategy: str, reason: str) -> dict[str, Any]:
    return {
        "decision_threshold": 0.5,
        "threshold_precision_floor": float(precision_floor),
        "threshold_constraint_met": False,
        "threshold_selection_metric": "default",
        "threshold_selection_strategy": reason,
        "threshold_strategy_requested": strategy,
        "threshold_val_precision": 0.0,
        "threshold_val_recall": 0.0,
        "threshold_val_f1": 0.0,
        "threshold_val_auc": float("nan"),
    }


def _candidate_record(
    threshold: float,
    metrics: dict[str, float],
    precision_floor: float,
    strategy: str,
) -> dict[str, Any]:
    return {
        "decision_threshold": float(threshold),
        "threshold_precision_floor": float(precision_floor),
        "threshold_constraint_met": bool(metrics["precision"] >= precision_floor),
        "threshold_strategy_requested": strategy,
        "threshold_val_precision": float(metrics["precision"]),
        "threshold_val_recall": float(metrics["recall"]),
        "threshold_val_f1": float(metrics["f1"]),
        "threshold_val_auc": float(metrics["auc"]),
    }


def select_decision_threshold(
    y_true: Iterable[int],
    y_score: Iterable[float],
    strategy: str = DEFAULT_THRESHOLD_STRATEGY,
    precision_floor: float = 0.30,
) -> dict[str, Any]:
    """Select a decision threshold from validation probabilities.

    Strategies
    ----------
    - `recall_with_precision_floor` (default, legacy): pick the highest-recall
      threshold whose validation precision is at least `precision_floor`.
      Falls back to the highest-F1 threshold when no candidate satisfies the
      floor; the constraint flag records whether the floor was actually met.
    - `f1_optimal`: pick the threshold that maximises validation F1. Useful
      for tiny test sets where a precision floor leaves no candidates and the
      legacy recall strategy ends up at a degenerate threshold.
    - `youden_j`: pick the threshold that maximises Youden's J statistic
      `recall + specificity - 1` on the ROC curve. Useful for ranking-quality
      comparisons because it is threshold-tuning-invariant to class prevalence.
    """
    if strategy not in THRESHOLD_STRATEGIES:
        raise ValueError(
            f"Unsupported threshold strategy: {strategy!r}. "
            f"Expected one of {THRESHOLD_STRATEGIES}."
        )

    y_true_list = [int(value) for value in y_true]
    score_list = [float(value) for value in y_score]
    if not y_true_list or not score_list or len(y_true_list) != len(score_list):
        return _empty_threshold_summary(precision_floor, strategy, "default_no_scores")

    thresholds = sorted(set(score_list + [0.0, 0.5, 1.0]))
    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = compute_metrics_at_threshold(y_true_list, score_list, threshold)
        candidates.append(_candidate_record(threshold, metrics, precision_floor, strategy))

    if strategy == "recall_with_precision_floor":
        constrained = [c for c in candidates if c["threshold_constraint_met"]]
        if constrained:
            selected = sorted(
                constrained,
                key=lambda row: (
                    row["threshold_val_recall"],
                    row["threshold_val_f1"],
                    row["threshold_val_precision"],
                    row["decision_threshold"],
                ),
                reverse=True,
            )[0]
            selected["threshold_selection_metric"] = "val_recall"
            selected["threshold_selection_strategy"] = "max_recall_with_precision_floor"
            return selected
        selected = sorted(
            candidates,
            key=lambda row: (
                row["threshold_val_f1"],
                row["threshold_val_recall"],
                row["threshold_val_precision"],
                row["decision_threshold"],
            ),
            reverse=True,
        )[0]
        selected["threshold_selection_metric"] = "val_f1"
        selected["threshold_selection_strategy"] = "fallback_max_f1_precision_floor_unmet"
        return selected

    if strategy == "f1_optimal":
        selected = sorted(
            candidates,
            key=lambda row: (
                row["threshold_val_f1"],
                row["threshold_val_recall"],
                row["threshold_val_precision"],
                row["decision_threshold"],
            ),
            reverse=True,
        )[0]
        selected["threshold_selection_metric"] = "val_f1"
        selected["threshold_selection_strategy"] = "max_f1"
        return selected

    if strategy == "youden_j":
        # Youden J = recall + specificity - 1; specificity = 1 - FPR.
        # Use sklearn roc_curve thresholds plus the candidate set so we can
        # report the exact threshold that maximises J.
        try:
            fpr, tpr, roc_thresholds = roc_curve(y_true_list, score_list)
        except ValueError:
            return _empty_threshold_summary(precision_floor, strategy, "default_no_two_classes")
        best_threshold: float | None = None
        best_j = float("-inf")
        for fp, tp, thr in zip(fpr.tolist(), tpr.tolist(), roc_thresholds.tolist()):
            if thr == float("inf") or thr == float("-inf") or thr != thr:  # filter NaN/inf
                continue
            j = float(tp) - float(fp)
            if j > best_j:
                best_j = j
                best_threshold = float(thr)
        if best_threshold is None:
            selected = candidates[len(candidates) // 2]
            selected["threshold_selection_metric"] = "val_youden_j"
            selected["threshold_selection_strategy"] = "fallback_median_threshold_youden_unavailable"
            return selected
        metrics = compute_metrics_at_threshold(y_true_list, score_list, best_threshold)
        selected = _candidate_record(best_threshold, metrics, precision_floor, strategy)
        selected["threshold_selection_metric"] = "val_youden_j"
        selected["threshold_selection_strategy"] = "max_youden_j"
        selected["threshold_youden_j"] = float(best_j)
        return selected

    # Should be unreachable thanks to the guard above.
    return _empty_threshold_summary(precision_floor, strategy, "default_unknown_strategy")


def select_recall_threshold(
    y_true: Iterable[int],
    y_score: Iterable[float],
    precision_floor: float = 0.30,
) -> dict[str, Any]:
    """Backwards-compatible alias for the recall-with-precision-floor strategy."""
    return select_decision_threshold(
        y_true,
        y_score,
        strategy="recall_with_precision_floor",
        precision_floor=precision_floor,
    )