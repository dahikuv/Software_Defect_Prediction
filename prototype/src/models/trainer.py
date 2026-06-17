"""Training helpers for baseline experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics import DEFAULT_THRESHOLD_STRATEGY, compute_classification_metrics, select_decision_threshold
from src.models.bundle import ModelBundle
from src.models.predict import _extract_positive_class_probability
from src.models.registry import get_model
from src.utils.provenance import artifact_uses_commit_text


def _validate_training_frame(X: pd.DataFrame, y: pd.Series, frame_name: str) -> None:
    if X.empty:
        raise ValueError(f"{frame_name} features are empty.")
    if X.shape[1] == 0:
        raise ValueError(f"{frame_name} has no usable feature columns.")
    if len(X) != len(y):
        raise ValueError(f"{frame_name} features and labels have different lengths.")
    if y.nunique(dropna=True) < 2:
        raise ValueError(f"{frame_name} labels must contain at least two classes.")


def configure_model_for_imbalance(model: Any, y_train: pd.Series) -> Any:
    """Set class-imbalance options when supported by the estimator."""
    if not hasattr(model, "get_params") or not hasattr(model, "set_params"):
        return model

    params = model.get_params()
    counts = y_train.value_counts().to_dict()
    positive = float(counts.get(1, 0))
    negative = float(counts.get(0, 0))
    scale_pos_weight = (negative / positive) if positive > 0 else 1.0

    updates: dict[str, Any] = {}
    if "scale_pos_weight" in params:
        updates["scale_pos_weight"] = scale_pos_weight
    elif "class_weight" in params:
        updates["class_weight"] = "balanced"
    if updates:
        model.set_params(**updates)
    return model


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    random_state: int = 42,
):
    """Fit a registered model and return it."""
    _validate_training_frame(X_train, y_train, "Training")
    model = get_model(model_name, random_state=random_state)
    model = configure_model_for_imbalance(model, y_train)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, threshold: float | None = None) -> dict[str, float]:
    """Evaluate a fitted model on a test set."""
    if X_test.empty:
        raise ValueError("Test features are empty.")
    if len(X_test) != len(y_test):
        raise ValueError("Test features and labels have different lengths.")

    y_score = None
    probability = _extract_positive_class_probability(model, X_test)
    if probability is not None:
        y_score = probability.to_numpy()
    elif hasattr(model, "decision_function"):
        y_score = np.asarray(model.decision_function(X_test))

    if threshold is not None and y_score is not None:
        y_pred = (np.asarray(y_score) >= float(threshold)).astype(int)
    else:
        y_pred = model.predict(X_test)

    return compute_classification_metrics(y_test, y_pred, y_score)


def resolve_decision_threshold(
    model: Any,
    X_val: pd.DataFrame | None,
    y_val: pd.Series | None,
    precision_floor: float = 0.30,
    threshold_strategy: str = DEFAULT_THRESHOLD_STRATEGY,
) -> dict[str, Any]:
    """Select the decision threshold from validation probabilities."""
    base_default: dict[str, Any] = {
        "decision_threshold": 0.5,
        "threshold_precision_floor": float(precision_floor),
        "threshold_constraint_met": False,
        "threshold_selection_metric": "default",
        "threshold_strategy_requested": threshold_strategy,
        "threshold_val_precision": float("nan"),
        "threshold_val_recall": float("nan"),
        "threshold_val_f1": float("nan"),
        "threshold_val_auc": float("nan"),
    }
    if X_val is None or y_val is None or X_val.empty or len(X_val) != len(y_val) or y_val.nunique(dropna=True) < 2:
        base_default["threshold_selection_strategy"] = "default_no_validation"
        return base_default

    probability = _extract_positive_class_probability(model, X_val)
    if probability is None:
        base_default["threshold_selection_strategy"] = "default_no_probability"
        return base_default
    return select_decision_threshold(
        y_val,
        probability.to_numpy(),
        strategy=threshold_strategy,
        precision_floor=precision_floor,
    )

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
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    threshold_precision_floor: float = 0.30,
    threshold_strategy: str = "recall_with_precision_floor",
    feature_preprocessor: Any | None = None,
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
    threshold_metadata = resolve_decision_threshold(
        model=model,
        X_val=X_val,
        y_val=y_val,
        precision_floor=threshold_precision_floor,
        threshold_strategy=threshold_strategy,
    )
    threshold = float(threshold_metadata["decision_threshold"])

    train_metrics = evaluate_model(model=model, X_test=X_train, y_test=y_train, threshold=threshold)
    val_metrics = (
        evaluate_model(model=model, X_test=X_val, y_test=y_val, threshold=threshold)
        if X_val is not None and y_val is not None and not X_val.empty and len(X_val) == len(y_val)
        else _empty_metrics()
    )
    test_metrics = evaluate_model(model=model, X_test=X_test, y_test=y_test, threshold=threshold)

    feature_family = str(feature_metadata.get("feature_family") or feature_metadata.get("feature_set") or "metrics_only")
    selection_data_source = (
        "validation"
        if threshold_metadata.get("threshold_selection_metric") != "default"
        else str(threshold_metadata.get("threshold_selection_strategy", "default_no_validation"))
    )
    result = {
        "dataset_name": dataset_name,
        "model": model_name,
        "feature_family": feature_family,
        "feature_set": feature_metadata.get("feature_set", feature_family),
        "text_feature_column": feature_metadata.get("text_feature_column", ""),
        "uses_commit_text": artifact_uses_commit_text(feature_metadata),
        "artifact_stage": feature_metadata.get("artifact_stage", "training"),
        "artifact_schema_version": feature_metadata.get("artifact_schema_version", "paper-v1"),
        "artifact_created_at": feature_metadata.get("artifact_created_at", ""),
        "artifact_group_key": feature_metadata.get("artifact_group_key", f"{dataset_name}::{model_name}"),
        "artifact_id": feature_metadata.get("artifact_id", f"{dataset_name}::{model_name}::training"),
        "source_results_table": feature_metadata.get("source_results_table", ""),
        "num_train_rows": int(len(X_train)),
        "num_val_rows": int(len(X_val)) if X_val is not None else 0,
        "num_test_rows": int(len(X_test)),
        "num_features": int(X_train.shape[1]),
        "selected_metrics": ",".join(feature_metadata.get("selected_metrics", [])),
        "missing_metrics": ",".join(feature_metadata.get("missing_metrics", [])),
        "dropped_all_nan_metrics": ",".join(feature_metadata.get("dropped_all_nan_metrics", [])),
        "metrics_num_features": int(feature_metadata.get("metrics_num_features", X_train.shape[1])),
        "tfidf_num_features": int(feature_metadata.get("tfidf_num_features", 0)),
        "tfidf_vocabulary_size": int(feature_metadata.get("tfidf_vocabulary_size", 0)),
        "has_commit_text": bool(feature_metadata.get("has_commit_text", False)),
        "selection_data_source": selection_data_source,
        "test_metrics_report_only": True,
        **threshold_metadata,
        **_prefix_metrics(train_metrics, "train_"),
        **_prefix_metrics(val_metrics, "val_"),
        **test_metrics,
        "gap_train_val_accuracy": _metric_delta(train_metrics["accuracy"], val_metrics["accuracy"]),
        "gap_train_val_precision": _metric_delta(train_metrics["precision"], val_metrics["precision"]),
        "gap_train_val_recall": _metric_delta(train_metrics["recall"], val_metrics["recall"]),
        "gap_train_val_f1": _metric_delta(train_metrics["f1"], val_metrics["f1"]),
        "gap_train_val_auc": _metric_delta(train_metrics["auc"], val_metrics["auc"]),
        "gap_train_test_accuracy": _metric_delta(train_metrics["accuracy"], test_metrics["accuracy"]),
        "gap_train_test_precision": _metric_delta(train_metrics["precision"], test_metrics["precision"]),
        "gap_train_test_recall": _metric_delta(train_metrics["recall"], test_metrics["recall"]),
        "gap_train_test_f1": _metric_delta(train_metrics["f1"], test_metrics["f1"]),
        "gap_train_test_auc": _metric_delta(train_metrics["auc"], test_metrics["auc"]),
    }
    bundle_metadata = {
        **result,
        "threshold_precision_floor": threshold_precision_floor,
        "threshold_strategy_requested": threshold_strategy,
        "feature_columns": list(X_train.columns),
    }
    bundle = ModelBundle(
        estimator=model,
        feature_columns=list(X_train.columns),
        decision_threshold=threshold,
        feature_family=feature_family,
        preprocessor=feature_preprocessor,
        metadata=bundle_metadata,
    )
    return bundle, result


def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _metric_delta(left: float, right: float) -> float:
    if pd.isna(left) or pd.isna(right):
        return float("nan")
    return float(left - right)


def _empty_metrics() -> dict[str, float]:
    return {
        "accuracy": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "auc": float("nan"),
    }
