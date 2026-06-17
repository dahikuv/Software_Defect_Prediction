"""Bootstrap CI95 and paired significance tests for model comparison.

This module is intentionally dependency-light: it relies on ``numpy`` and
``scipy.stats`` which already ship with the project. The helpers operate on
1D arrays of ground-truth labels and predicted probabilities/scores so they
can be reused for both PROMISE and commit-level (JITLine) test sets.

Significance design notes
-------------------------
- ``bootstrap_ci_metric`` returns a percentile bootstrap CI for a single model
  on a single test set. Use it to attach uncertainty to point estimates.
- ``bootstrap_paired_delta`` returns a percentile bootstrap CI for the paired
  delta (B - A) on the same test rows. The CI is the right object to read for
  effect size; we deliberately do NOT compute a p-value from these bootstrap
  deltas because the Wilcoxon-on-bootstrap-deltas trick treats every bootstrap
  draw as an independent observation and shrinks the p-value with ``n_iter``,
  which is statistically invalid. Use ``paired_permutation_test`` or
  ``delong_test_auc`` for the actual p-value.
- ``paired_permutation_test`` runs a label-free paired permutation that swaps
  the score assignments (A vs B) row-by-row and recomputes the metric delta.
  This is a proper exchangeable test under the null "the two models are
  indistinguishable on this test set".
- ``delong_test_auc`` is a fast DeLong (Sun and Xu, 2014) test for paired ROC
  AUCs. Use it when the metric of interest is AUC.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics import compute_metrics_at_threshold

DEFAULT_BOOTSTRAP_ITERS = 1000
DEFAULT_PERMUTATION_ITERS = 1000
DEFAULT_ALPHA = 0.05
THRESHOLD_METRICS = ("accuracy", "precision", "recall", "f1")
RANK_METRICS = ("auc",)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _metric_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    metric_name: str,
) -> float:
    if metric_name == "auc":
        return _safe_auc(y_true, y_score)
    metrics = compute_metrics_at_threshold(y_true.tolist(), y_score.tolist(), float(threshold))
    return float(metrics.get(metric_name, float("nan")))


def bootstrap_ci_metric(
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float,
    metric_name: str,
    n_iter: int = DEFAULT_BOOTSTRAP_ITERS,
    seed: int = 42,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Percentile bootstrap CI for one classification metric."""
    if metric_name not in THRESHOLD_METRICS and metric_name not in RANK_METRICS:
        raise ValueError(f"Unsupported metric for bootstrap CI: {metric_name}")

    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_score_arr = np.asarray(list(y_score), dtype=float)
    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("y_true and y_score must share the same length")
    n = y_true_arr.shape[0]

    nan = float("nan")
    if n == 0:
        return {
            "metric": metric_name,
            "point": nan,
            "ci_low": nan,
            "ci_high": nan,
            "n_iter_effective": 0,
            "n_samples": 0,
            "alpha": float(alpha),
        }

    point = _metric_at_threshold(y_true_arr, y_score_arr, threshold, metric_name)
    rng = np.random.default_rng(int(seed))
    samples: list[float] = []
    for _ in range(int(n_iter)):
        idx = rng.integers(0, n, size=n)
        bs_true = y_true_arr[idx]
        bs_score = y_score_arr[idx]
        value = _metric_at_threshold(bs_true, bs_score, threshold, metric_name)
        if np.isfinite(value):
            samples.append(value)

    if not samples:
        return {
            "metric": metric_name,
            "point": point,
            "ci_low": nan,
            "ci_high": nan,
            "n_iter_effective": 0,
            "n_samples": int(n),
            "alpha": float(alpha),
        }

    arr = np.asarray(samples, dtype=float)
    return {
        "metric": metric_name,
        "point": float(point),
        "ci_low": float(np.quantile(arr, alpha / 2.0)),
        "ci_high": float(np.quantile(arr, 1.0 - alpha / 2.0)),
        "n_iter_effective": int(arr.size),
        "n_samples": int(n),
        "alpha": float(alpha),
    }


def bootstrap_paired_delta(
    y_true: Iterable[int],
    score_a: Iterable[float],
    score_b: Iterable[float],
    threshold_a: float,
    threshold_b: float,
    metric_name: str,
    n_iter: int = DEFAULT_BOOTSTRAP_ITERS,
    seed: int = 42,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Paired percentile bootstrap for the (B minus A) delta of one metric.

    Returns the bootstrap distribution of the delta and its percentile CI.
    Does NOT return a p-value: use ``paired_permutation_test`` (any metric)
    or ``delong_test_auc`` (AUC) for hypothesis testing.
    """
    if metric_name not in THRESHOLD_METRICS and metric_name not in RANK_METRICS:
        raise ValueError(f"Unsupported metric for paired bootstrap: {metric_name}")

    y_true_arr = np.asarray(list(y_true), dtype=int)
    a = np.asarray(list(score_a), dtype=float)
    b = np.asarray(list(score_b), dtype=float)
    n = y_true_arr.shape[0]

    nan = float("nan")
    if n == 0 or a.shape[0] != n or b.shape[0] != n:
        return {
            "metric": metric_name,
            "delta": nan,
            "delta_ci_low": nan,
            "delta_ci_high": nan,
            "n_iter_effective": 0,
            "n_samples": int(n),
            "point_a": nan,
            "point_b": nan,
        }

    point_a = _metric_at_threshold(y_true_arr, a, threshold_a, metric_name)
    point_b = _metric_at_threshold(y_true_arr, b, threshold_b, metric_name)
    point_delta = float(point_b - point_a) if np.isfinite(point_a) and np.isfinite(point_b) else nan

    rng = np.random.default_rng(int(seed))
    deltas: list[float] = []
    for _ in range(int(n_iter)):
        idx = rng.integers(0, n, size=n)
        bs_true = y_true_arr[idx]
        bs_a = a[idx]
        bs_b = b[idx]
        ma = _metric_at_threshold(bs_true, bs_a, threshold_a, metric_name)
        mb = _metric_at_threshold(bs_true, bs_b, threshold_b, metric_name)
        if np.isfinite(ma) and np.isfinite(mb):
            deltas.append(mb - ma)

    if not deltas:
        return {
            "metric": metric_name,
            "delta": point_delta,
            "delta_ci_low": nan,
            "delta_ci_high": nan,
            "n_iter_effective": 0,
            "n_samples": int(n),
            "point_a": float(point_a) if np.isfinite(point_a) else nan,
            "point_b": float(point_b) if np.isfinite(point_b) else nan,
        }

    arr = np.asarray(deltas, dtype=float)
    return {
        "metric": metric_name,
        "delta": point_delta,
        "delta_ci_low": float(np.quantile(arr, alpha / 2.0)),
        "delta_ci_high": float(np.quantile(arr, 1.0 - alpha / 2.0)),
        "n_iter_effective": int(arr.size),
        "n_samples": int(n),
        "point_a": float(point_a) if np.isfinite(point_a) else nan,
        "point_b": float(point_b) if np.isfinite(point_b) else nan,
    }


def paired_permutation_test(
    y_true: Iterable[int],
    score_a: Iterable[float],
    score_b: Iterable[float],
    threshold_a: float,
    threshold_b: float,
    metric_name: str,
    n_iter: int = DEFAULT_PERMUTATION_ITERS,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired permutation test for the (B - A) delta of one metric.

    Under the null hypothesis "the two score columns are exchangeable", we
    randomly swap A and B per-row and recompute the metric delta. The
    two-sided p-value is the fraction of permuted deltas whose magnitude is
    at least as large as the observed delta. Thresholds for A and B are also
    swapped together with the scores so the binarisation is consistent.
    """
    if metric_name not in THRESHOLD_METRICS and metric_name not in RANK_METRICS:
        raise ValueError(f"Unsupported metric for permutation test: {metric_name}")

    y_true_arr = np.asarray(list(y_true), dtype=int)
    a = np.asarray(list(score_a), dtype=float)
    b = np.asarray(list(score_b), dtype=float)
    n = y_true_arr.shape[0]

    nan = float("nan")
    if n == 0 or a.shape[0] != n or b.shape[0] != n:
        return {
            "metric": metric_name,
            "observed_delta": nan,
            "p_value": nan,
            "n_iter_effective": 0,
            "n_samples": int(n),
            "test": "paired_permutation",
        }

    point_a = _metric_at_threshold(y_true_arr, a, threshold_a, metric_name)
    point_b = _metric_at_threshold(y_true_arr, b, threshold_b, metric_name)
    if not (np.isfinite(point_a) and np.isfinite(point_b)):
        return {
            "metric": metric_name,
            "observed_delta": nan,
            "p_value": nan,
            "n_iter_effective": 0,
            "n_samples": int(n),
            "test": "paired_permutation",
        }
    observed_delta = float(point_b - point_a)
    abs_observed = abs(observed_delta)

    rng = np.random.default_rng(int(seed))
    n_extreme = 0
    n_effective = 0
    for _ in range(int(n_iter)):
        swap = rng.integers(0, 2, size=n).astype(bool)
        scores_a_perm = np.where(swap, b, a)
        scores_b_perm = np.where(swap, a, b)
        # Thresholds also flip with the scores so binarisation matches the source model.
        thr_a_perm = np.where(swap, threshold_b, threshold_a)
        thr_b_perm = np.where(swap, threshold_a, threshold_b)
        if metric_name == "auc":
            ma = _safe_auc(y_true_arr, scores_a_perm)
            mb = _safe_auc(y_true_arr, scores_b_perm)
        else:
            pred_a = (scores_a_perm >= thr_a_perm).astype(int)
            pred_b = (scores_b_perm >= thr_b_perm).astype(int)
            metrics_a = compute_metrics_at_threshold(y_true_arr.tolist(), pred_a.tolist(), 0.5)
            metrics_b = compute_metrics_at_threshold(y_true_arr.tolist(), pred_b.tolist(), 0.5)
            ma = float(metrics_a.get(metric_name, nan))
            mb = float(metrics_b.get(metric_name, nan))
        if not (np.isfinite(ma) and np.isfinite(mb)):
            continue
        n_effective += 1
        if abs(mb - ma) >= abs_observed:
            n_extreme += 1

    if n_effective == 0:
        return {
            "metric": metric_name,
            "observed_delta": observed_delta,
            "p_value": nan,
            "n_iter_effective": 0,
            "n_samples": int(n),
            "test": "paired_permutation",
        }

    p_value = float((n_extreme + 1) / (n_effective + 1))
    return {
        "metric": metric_name,
        "observed_delta": observed_delta,
        "p_value": p_value,
        "n_iter_effective": int(n_effective),
        "n_samples": int(n),
        "test": "paired_permutation",
    }


def delong_test_auc(
    y_true: Iterable[int],
    score_a: Iterable[float],
    score_b: Iterable[float],
) -> dict[str, Any]:
    """Fast DeLong test for paired ROC AUCs (Sun and Xu, 2014)."""
    y_true_arr = np.asarray(list(y_true), dtype=int)
    a = np.asarray(list(score_a), dtype=float)
    b = np.asarray(list(score_b), dtype=float)

    nan = float("nan")
    pos_idx = np.where(y_true_arr == 1)[0]
    neg_idx = np.where(y_true_arr == 0)[0]
    n_pos = int(pos_idx.size)
    n_neg = int(neg_idx.size)
    if n_pos == 0 or n_neg == 0:
        return {
            "auc_a": nan,
            "auc_b": nan,
            "auc_diff": nan,
            "z": nan,
            "p_value": nan,
            "test": "delong",
            "n_pos": n_pos,
            "n_neg": n_neg,
        }

    def _components(scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        pos = scores[pos_idx]
        neg = scores[neg_idx]
        # Vectorised psi via broadcasting; memory cost O(n_pos * n_neg)
        diff = pos[:, None] - neg[None, :]
        psi = (diff > 0).astype(float) + 0.5 * (diff == 0).astype(float)
        v10 = psi.mean(axis=1)
        v01 = psi.mean(axis=0)
        return float(v10.mean()), v10, v01

    auc_a, v10_a, v01_a = _components(a)
    auc_b, v10_b, v01_b = _components(b)

    if n_pos > 1:
        s10 = np.cov(np.vstack([v10_a, v10_b]), ddof=1)
    else:
        s10 = np.zeros((2, 2), dtype=float)
    if n_neg > 1:
        s01 = np.cov(np.vstack([v01_a, v01_b]), ddof=1)
    else:
        s01 = np.zeros((2, 2), dtype=float)

    var_a = s10[0, 0] / n_pos + s01[0, 0] / n_neg
    var_b = s10[1, 1] / n_pos + s01[1, 1] / n_neg
    cov_ab = s10[0, 1] / n_pos + s01[0, 1] / n_neg
    var_diff = float(var_a + var_b - 2.0 * cov_ab)
    diff = float(auc_b - auc_a)

    if not np.isfinite(var_diff) or var_diff <= 0:
        return {
            "auc_a": float(auc_a),
            "auc_b": float(auc_b),
            "auc_diff": diff,
            "z": nan,
            "p_value": nan,
            "test": "delong",
            "n_pos": n_pos,
            "n_neg": n_neg,
        }

    z = diff / float(np.sqrt(var_diff))
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "auc_diff": diff,
        "z": float(z),
        "p_value": p,
        "test": "delong",
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def benjamini_hochberg(p_values: Iterable[float], alpha: float = DEFAULT_ALPHA) -> dict[str, Any]:
    """Benjamini-Hochberg FDR correction over a family of p-values.

    Returns the adjusted p-values (same order as the input) and a boolean
    rejection mask at level ``alpha``. NaN inputs are carried through as NaN
    and excluded from the ranking so they never consume a comparison slot.
    """
    raw = np.asarray(list(p_values), dtype=float)
    n = raw.size
    adjusted = np.full(n, np.nan, dtype=float)
    rejected = np.zeros(n, dtype=bool)

    finite_idx = np.where(np.isfinite(raw))[0]
    m = int(finite_idx.size)
    if m == 0:
        return {"p_adjusted": adjusted.tolist(), "rejected": rejected.tolist(), "n_tests": 0}

    finite_p = raw[finite_idx]
    order = np.argsort(finite_p, kind="mergesort")
    ranked = finite_p[order]
    ranks = np.arange(1, m + 1, dtype=float)

    # Standard BH step-up: adjusted_(i) = min over k>=i of (m/k) * p_(k).
    scaled = ranked * m / ranks
    adjusted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted_finite = np.empty(m, dtype=float)
    adjusted_finite[order] = adjusted_sorted
    adjusted[finite_idx] = adjusted_finite
    rejected[finite_idx] = adjusted_finite <= float(alpha)

    return {"p_adjusted": adjusted.tolist(), "rejected": rejected.tolist(), "n_tests": m}


def wilcoxon_paired_test(deltas: Iterable[float]) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank test on a 1D array of paired deltas.

    Use only with cross-dataset deltas (one delta per dataset), not with
    bootstrap-derived deltas on a single test set.
    """
    arr = np.asarray(list(deltas), dtype=float)
    arr = arr[np.isfinite(arr)]
    nan = float("nan")
    if arr.size == 0:
        return {"statistic": nan, "p_value": nan, "n_pairs": 0, "test": "wilcoxon_signed_rank"}
    try:
        stat, p_value = stats.wilcoxon(arr, zero_method="zsplit", alternative="two-sided")
        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "n_pairs": int(arr.size),
            "test": "wilcoxon_signed_rank",
        }
    except ValueError:
        return {"statistic": nan, "p_value": nan, "n_pairs": int(arr.size), "test": "wilcoxon_signed_rank"}