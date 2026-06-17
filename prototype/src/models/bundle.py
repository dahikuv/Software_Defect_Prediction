"""Serializable model bundle for aligned preprocessing and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelBundle:
    """Fitted estimator plus feature preprocessing and decision metadata."""

    estimator: Any
    feature_columns: list[str]
    decision_threshold: float | None = None
    feature_family: str = "metrics_only"
    preprocessor: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is not None and hasattr(self.preprocessor, "transform"):
            return self.preprocessor.transform(X)
        if self.preprocessor is not None and hasattr(self.preprocessor, "selected_metrics"):
            from src.features.metrics_features import transform_metrics_features

            return transform_metrics_features(X, self.preprocessor)

        prepared = X.copy()
        for column in self.feature_columns:
            if column not in prepared.columns:
                prepared[column] = 0.0
        return prepared[self.feature_columns].copy()

    def predict_proba(self, X: pd.DataFrame):
        return self.estimator.predict_proba(self.transform_features(X))

    def decision_function(self, X: pd.DataFrame):
        return self.estimator.decision_function(self.transform_features(X))

    def predict(self, X: pd.DataFrame):
        prepared = self.transform_features(X)
        if self.decision_threshold is None:
            return self.estimator.predict(prepared)

        if not hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict(prepared)

        proba = np.asarray(self.estimator.predict_proba(prepared))
        if proba.ndim != 2 or proba.shape[0] != len(prepared) or proba.shape[1] == 0:
            return self.estimator.predict(prepared)

        if proba.shape[1] == 1:
            positive_index = 0
        else:
            class_labels = list(getattr(self.estimator, "classes_", []))
            if len(class_labels) == proba.shape[1] and 1 in class_labels:
                positive_index = class_labels.index(1)
            else:
                positive_index = 1
            positive_index = min(max(int(positive_index), 0), proba.shape[1] - 1)

        positive_scores = proba[:, positive_index]
        return (positive_scores >= float(self.decision_threshold)).astype(int)
