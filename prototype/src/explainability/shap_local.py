"""Local explainability analysis helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import shap

from src.utils.io import write_csv
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _ensure_output_dir(output_dir: str | Path) -> Path:
    """Create the output directory if needed and return it as a Path."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_single_row_values(shap_values: Any) -> Any:
    """Normalize a one-row SHAP output to a 1D feature vector."""
    values = getattr(shap_values, "values", shap_values)

    if isinstance(values, list):
        if len(values) == 1:
            values = values[0]
        else:
            values = values[-1]

    if getattr(values, "ndim", 1) == 3:
        values = values[:, :, -1]
    if getattr(values, "ndim", 1) == 2:
        values = values[0]

    return values


def _compute_true_local_shap(model: Any, X_reference: pd.DataFrame, X_row: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Compute a true SHAP local explanation for one row."""
    logger.info(
        "[%s] building local TreeExplainer with reference rows=%s cols=%s",
        dataset_name,
        len(X_reference),
        X_reference.shape[1],
    )
    explainer = shap.TreeExplainer(model, data=X_reference, feature_perturbation="interventional")
    logger.info("[%s] computing local true SHAP values", dataset_name)
    shap_values = explainer.shap_values(X_row)
    row_values = _normalize_single_row_values(shap_values)

    return pd.DataFrame(
        {
            "feature": X_row.columns,
            "feature_value": X_row.iloc[0].values,
            "shap_value": row_values,
        }
    ).sort_values("shap_value", key=lambda s: s.abs(), ascending=False)


def _compute_approx_local_contributions(model: Any, X_reference: pd.DataFrame, X_row: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Build a stable local contribution table without relying on SHAP runtime."""
    baseline = X_reference.median(axis=0)
    row = X_row.iloc[0]

    if hasattr(model, "predict_proba"):
        logger.info("[%s] using predict_proba perturbation fallback for local explainability", dataset_name)
        baseline_score = model.predict_proba(pd.DataFrame([baseline], columns=X_row.columns))[0, 1]
        contributions = []
        for feature in X_row.columns:
            perturbed = baseline.copy()
            perturbed[feature] = row[feature]
            score = model.predict_proba(pd.DataFrame([perturbed], columns=X_row.columns))[0, 1]
            contributions.append(score - baseline_score)
    else:
        logger.info("[%s] using centered feature delta fallback for local explainability", dataset_name)
        contributions = (row - baseline).to_list()

    return pd.DataFrame(
        {
            "feature": X_row.columns,
            "feature_value": row.values,
            "contribution": contributions,
        }
    ).sort_values("contribution", key=lambda s: s.abs(), ascending=False)


def run_local_shap(
    model: Any,
    X_reference: pd.DataFrame,
    X_row: pd.DataFrame,
    output_dir: str | Path,
    dataset_name: str,
    row_label: str = "sample_0",
    mode: str = "true_shap",
    allow_fallback: bool = True,
) -> dict[str, str]:
    """Compute and save a local explainability table for one row."""
    output_path = _ensure_output_dir(output_dir)
    local_csv = output_path / f"{dataset_name}_{row_label}_shap_local.csv"

    if mode == "true_shap":
        try:
            local_df = _compute_true_local_shap(model, X_reference, X_row, dataset_name)
            write_csv(local_df, local_csv)
            return {"local_csv": str(local_csv)}
        except Exception as exc:
            logger.warning("[%s] true local SHAP failed: %s", dataset_name, exc)
            if not allow_fallback:
                raise

    local_df = _compute_approx_local_contributions(model, X_reference, X_row, dataset_name)
    write_csv(local_df, local_csv)
    return {"local_csv": str(local_csv)}


__all__ = ["run_local_shap"]
