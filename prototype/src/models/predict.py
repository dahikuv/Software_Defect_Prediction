"""Prediction helpers for saved model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def load_model(path: str | Path):
    """Load a serialized model."""
    return joblib.load(path)


def _extract_positive_class_probability(model: Any, X: pd.DataFrame) -> pd.Series | None:
    """Return a stable probability series when the estimator supports it."""
    if not hasattr(model, "predict_proba"):
        return None

    proba = model.predict_proba(X)
    if proba is None:
        return None

    proba_array = np.asarray(proba)
    if proba_array.ndim != 2 or proba_array.shape[0] != len(X):
        return None
    if proba_array.shape[1] == 0:
        return None

    if proba_array.shape[1] == 1:
        return pd.Series(proba_array[:, 0], index=X.index, name="probability")

    class_labels = list(getattr(model, "classes_", []))
    if len(class_labels) == proba_array.shape[1] and 1 in class_labels:
        positive_index = class_labels.index(1)
    else:
        positive_index = 1
    positive_index = min(max(int(positive_index), 0), proba_array.shape[1] - 1)
    return pd.Series(proba_array[:, positive_index], index=X.index, name="probability")


def predict_with_model(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Return predictions and optional probabilities."""
    if X.empty:
        raise ValueError("Prediction input is empty.")

    output = pd.DataFrame(index=X.index)
    output["prediction"] = model.predict(X)
    probability = _extract_positive_class_probability(model, X)
    if probability is not None:
        output["probability"] = probability
    return output
