from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.evaluation.metrics import select_recall_threshold
from src.features.metrics_features import fit_metrics_feature_spec, transform_metrics_features
from src.models.bundle import ModelBundle
from src.models.predict import predict_with_model
from src.models.trainer import configure_model_for_imbalance
from src.utils.coercion import coerce_bool


class FakeProbabilityModel:
    classes_ = np.array([0, 1])

    def __init__(self, scores: list[float]) -> None:
        self.scores = np.asarray(scores, dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        scores = self.scores[: len(X)]
        return np.column_stack([1.0 - scores, scores])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        scores = self.scores[: len(X)]
        return (scores >= 0.5).astype(int)

class FakeDualImbalanceModel:
    def __init__(self) -> None:
        self.params = {"class_weight": None, "scale_pos_weight": None}
        self.updates: dict[str, object] = {}

    def get_params(self) -> dict[str, object]:
        return dict(self.params)

    def set_params(self, **params: object):
        self.updates.update(params)
        self.params.update(params)
        return self


class ThresholdSelectionTests(unittest.TestCase):
    def test_selects_max_recall_threshold_with_precision_floor(self) -> None:
        selected = select_recall_threshold(
            y_true=[1, 1, 0, 0],
            y_score=[0.9, 0.4, 0.8, 0.1],
            precision_floor=0.5,
        )

        self.assertTrue(selected["threshold_constraint_met"])
        self.assertEqual(selected["threshold_selection_strategy"], "max_recall_with_precision_floor")
        self.assertAlmostEqual(selected["decision_threshold"], 0.4)
        self.assertAlmostEqual(selected["threshold_val_recall"], 1.0)
        self.assertGreaterEqual(selected["threshold_val_precision"], 0.5)

    def test_falls_back_to_f1_when_precision_floor_is_unmet(self) -> None:
        selected = select_recall_threshold(
            y_true=[1, 0, 0, 0],
            y_score=[0.6, 0.9, 0.8, 0.7],
            precision_floor=0.3,
        )

        self.assertFalse(selected["threshold_constraint_met"])
        self.assertEqual(selected["threshold_selection_strategy"], "fallback_max_f1_precision_floor_unmet")
        self.assertEqual(selected["threshold_selection_metric"], "val_f1")
        self.assertAlmostEqual(selected["decision_threshold"], 0.6)


class ThresholdStrategyTests(unittest.TestCase):
    def test_f1_optimal_picks_threshold_with_max_f1(self) -> None:
        from src.evaluation.metrics import select_decision_threshold

        selected = select_decision_threshold(
            y_true=[1, 1, 0, 0, 0],
            y_score=[0.9, 0.6, 0.55, 0.4, 0.1],
            strategy="f1_optimal",
        )
        self.assertEqual(selected["threshold_selection_metric"], "val_f1")
        self.assertEqual(selected["threshold_selection_strategy"], "max_f1")
        self.assertEqual(selected["threshold_strategy_requested"], "f1_optimal")
        self.assertGreaterEqual(selected["threshold_val_f1"], 0.66)

    def test_youden_j_picks_threshold_maximising_j(self) -> None:
        from src.evaluation.metrics import select_decision_threshold

        selected = select_decision_threshold(
            y_true=[1, 1, 1, 0, 0, 0],
            y_score=[0.9, 0.7, 0.4, 0.5, 0.3, 0.1],
            strategy="youden_j",
        )
        self.assertEqual(selected["threshold_selection_metric"], "val_youden_j")
        self.assertEqual(selected["threshold_selection_strategy"], "max_youden_j")
        self.assertEqual(selected["threshold_strategy_requested"], "youden_j")
        self.assertGreaterEqual(selected["threshold_youden_j"], 0.65)

    def test_select_decision_threshold_rejects_unknown_strategy(self) -> None:
        from src.evaluation.metrics import select_decision_threshold

        with self.assertRaises(ValueError):
            select_decision_threshold([0, 1], [0.1, 0.9], strategy="bogus")

class PredictionThresholdTests(unittest.TestCase):
    def test_predict_with_model_uses_supplied_threshold(self) -> None:
        X = pd.DataFrame({"loc": [10, 20, 30]})
        model = FakeProbabilityModel([0.2, 0.45, 0.7])

        predictions = predict_with_model(model, X, threshold=0.4)

        self.assertListEqual(predictions["prediction"].tolist(), [0, 1, 1])
        self.assertListEqual(predictions["probability"].round(2).tolist(), [0.2, 0.45, 0.7])
        self.assertTrue((predictions["decision_threshold"] == 0.4).all())

    def test_model_bundle_uses_saved_threshold_and_train_feature_spec(self) -> None:
        train_df = pd.DataFrame({"loc": [10.0, 20.0, None], "label": [0, 1, 0]})
        spec = fit_metrics_feature_spec(train_df, ["loc"])
        X = pd.DataFrame({"loc": [None, 30.0]})
        model = FakeProbabilityModel([0.45, 0.55])
        bundle = ModelBundle(
            estimator=model,
            feature_columns=["loc"],
            decision_threshold=0.5,
            preprocessor=spec,
        )

        transformed = transform_metrics_features(X, spec)
        predictions = predict_with_model(bundle, X)

        self.assertListEqual(transformed["loc"].tolist(), [15.0, 30.0])
        self.assertListEqual(predictions["prediction"].tolist(), [0, 1])
        self.assertTrue((predictions["decision_threshold"] == 0.5).all())


class CoercionTests(unittest.TestCase):
    def test_coerce_bool_handles_csv_strings(self) -> None:
        self.assertFalse(coerce_bool("False"))
        self.assertFalse(coerce_bool("0"))
        self.assertFalse(coerce_bool(""))
        self.assertTrue(coerce_bool("true"))
        self.assertTrue(coerce_bool(1))

class ImbalanceConfigurationTests(unittest.TestCase):
    def test_scale_pos_weight_takes_precedence_over_class_weight(self) -> None:
        model = FakeDualImbalanceModel()
        configured = configure_model_for_imbalance(model, pd.Series([0, 0, 0, 1]))

        self.assertIs(configured, model)
        self.assertEqual(model.updates, {"scale_pos_weight": 3.0})


if __name__ == "__main__":
    unittest.main()
