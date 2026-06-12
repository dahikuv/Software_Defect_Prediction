"""Global SHAP analysis helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.utils.io import write_csv
from src.utils.logging import get_logger

matplotlib.use("Agg")
logger = get_logger(__name__)


def _ensure_output_dir(output_dir: str | Path) -> Path:
    """Create the output directory if needed and return it as a Path."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_shap_values(shap_values: Any) -> Any:
    """Normalize SHAP outputs to a 2D feature matrix."""
    values = getattr(shap_values, "values", shap_values)

    if isinstance(values, list):
        if len(values) == 1:
            values = values[0]
        else:
            values = values[-1]

    if getattr(values, "ndim", 1) == 3:
        values = values[:, :, -1]

    return values


def _compute_true_shap_values(model: Any, X_background: pd.DataFrame, X_explain: pd.DataFrame, dataset_name: str) -> Any:
    """Compute true SHAP values for tree-based models."""
    logger.info(
        "[%s] building TreeExplainer with background rows=%s cols=%s",
        dataset_name,
        len(X_background),
        X_background.shape[1],
    )
    explainer = shap.TreeExplainer(model, data=X_background, feature_perturbation="interventional")
    logger.info(
        "[%s] computing true SHAP values for explain rows=%s cols=%s",
        dataset_name,
        len(X_explain),
        X_explain.shape[1],
    )
    return explainer.shap_values(X_explain)


def _compute_approx_importance(model: Any, X_explain: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Build a stable approximation when true SHAP is unavailable."""
    if hasattr(model, "feature_importances_"):
        logger.info("[%s] using model.feature_importances_ fallback", dataset_name)
        importances = model.feature_importances_
    else:
        logger.info("[%s] using feature variance fallback", dataset_name)
        importances = X_explain.var(axis=0).to_numpy()

    return pd.DataFrame(
        {
            "feature": X_explain.columns,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)


def _compute_true_shap_summary(X_explain: pd.DataFrame, shap_values: Any) -> pd.DataFrame:
    """Aggregate per-feature SHAP importance from true SHAP outputs."""
    values = _normalize_shap_values(shap_values)
    mean_abs_values = abs(values).mean(axis=0)
    return pd.DataFrame(
        {
            "feature": X_explain.columns,
            "mean_abs_shap": mean_abs_values,
        }
    ).sort_values("mean_abs_shap", ascending=False)


def _save_summary_plot(shap_values: Any, X_explain: pd.DataFrame, plot_path: Path) -> str:
    """Save a SHAP summary plot."""
    values = _normalize_shap_values(shap_values)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(values, X_explain, show=False)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    return str(plot_path)


def _save_importance_plot(summary_df: pd.DataFrame, plot_path: Path, value_column: str) -> str:
    """Save a fallback bar chart for approximate explainability."""
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df["feature"], summary_df[value_column])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(value_column)
    plt.title("Global Feature Importance")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    return str(plot_path)


def run_global_shap(
    model: Any,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    output_dir: str | Path,
    dataset_name: str,
    mode: str = "true_shap",
    enable_plots: bool = False,
    allow_fallback: bool = True,
) -> dict[str, str]:
    """Compute and save global explainability outputs for one fitted model."""
    output_path = _ensure_output_dir(output_dir)
    summary_csv = output_path / f"{dataset_name}_shap_global_summary.csv"
    importance_path = output_path / f"{dataset_name}_shap_importance.csv"
    plot_path = output_path / f"{dataset_name}_shap_summary.png"

    use_true_shap = mode == "true_shap"

    if use_true_shap:
        try:
            shap_values = _compute_true_shap_values(model, X_background, X_explain, dataset_name)
            summary_df = _compute_true_shap_summary(X_explain, shap_values)
            write_csv(summary_df, summary_csv)
            write_csv(summary_df, importance_path)

            outputs = {
                "summary_csv": str(summary_csv),
                "importance_csv": str(importance_path),
            }
            if enable_plots:
                outputs["plot_path"] = _save_summary_plot(shap_values, X_explain, plot_path)
            return outputs
        except Exception as exc:
            logger.warning("[%s] true SHAP failed: %s", dataset_name, exc)
            if not allow_fallback:
                raise

    summary_df = _compute_approx_importance(model, X_explain, dataset_name)
    write_csv(summary_df, summary_csv)
    write_csv(summary_df, importance_path)

    outputs = {
        "summary_csv": str(summary_csv),
        "importance_csv": str(importance_path),
    }
    if enable_plots:
        outputs["plot_path"] = _save_importance_plot(summary_df, plot_path, summary_df.columns[1])
    return outputs
