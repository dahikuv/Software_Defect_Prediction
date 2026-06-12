"""Training helpers for baseline experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_classification_metrics
from src.models.predict import _extract_positive_class_probability
from src.models.registry import get_model


def _validate_training_frame(X: pd.DataFrame, y: pd.Series, frame_name: str) -> None:
    if X.empty:
        raise ValueError(f"{frame_name} features are empty.")
    if X.shape[1] == 0:
        raise ValueError(f"{frame_name} has no usable feature columns.")
    if len(X) != len(y):
        raise ValueError(f"{frame_name} features and labels have different lengths.")
    if y.nunique(dropna=True) < 2:
        raise ValueError(f"{frame_name} labels must contain at least two classes.")


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    random_state: int = 42,
):
    """Fit a registered model and return it."""
    _validate_training_frame(X_train, y_train, "Training")
    model = get_model(model_name, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate a fitted model on a test set."""
    if X_test.empty:
        raise ValueError("Test features are empty.")
    if len(X_test) != len(y_test):
        raise ValueError("Test features and labels have different lengths.")

    y_pred = model.predict(X_test)

    y_score = None
    probability = _extract_positive_class_probability(model, X_test)
    if probability is not None:
        y_score = probability.to_numpy()
    elif hasattr(model, "decision_function"):
        y_score = np.asarray(model.decision_function(X_test))

    return compute_classification_metrics(y_test, y_pred, y_score)


def save_model(model: Any, path: str | Path) -> None:
    """Persist a fitted model artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def train_and_evaluate_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    dataset_name: str,
    random_state: int = 42,
    feature_metadata: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Train one model and return both the fitted artifact and a result record.

    The result record is intended to be written directly into a results table.
    """
    feature_metadata = feature_metadata or {}
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        model_name=model_name,
        random_state=random_state,
    )
    metrics = evaluate_model(model=model, X_test=X_test, y_test=y_test)

    feature_family = str(feature_metadata.get("feature_family") or feature_metadata.get("feature_set") or "metrics_only")
    result = {
        "dataset_name": dataset_name,
        "model": model_name,
        "feature_family": feature_family,
        "feature_set": feature_metadata.get("feature_set", feature_family),
        "text_feature_column": feature_metadata.get("text_feature_column", ""),
        "uses_commit_text": bool(feature_metadata.get("uses_commit_text", feature_family in {"metrics_plus_commit_text", "metrics_plus_text", "hybrid"})),
        "artifact_stage": feature_metadata.get("artifact_stage", "training"),
        "artifact_schema_version": feature_metadata.get("artifact_schema_version", "paper-v1"),
        "artifact_created_at": feature_metadata.get("artifact_created_at", ""),
        "artifact_group_key": feature_metadata.get("artifact_group_key", f"{dataset_name}::{model_name}"),
        "artifact_id": feature_metadata.get("artifact_id", f"{dataset_name}::{model_name}::training"),
        "source_results_table": feature_metadata.get("source_results_table", ""),
        "num_train_rows": int(len(X_train)),
        "num_test_rows": int(len(X_test)),
        "num_features": int(X_train.shape[1]),
        "selected_metrics": ",".join(feature_metadata.get("selected_metrics", [])),
        "missing_metrics": ",".join(feature_metadata.get("missing_metrics", [])),
        "dropped_all_nan_metrics": ",".join(feature_metadata.get("dropped_all_nan_metrics", [])),
        **metrics,
    }
    return model, result
