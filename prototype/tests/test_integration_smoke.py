from __future__ import annotations

import unittest
from pathlib import Path

import joblib
import pandas as pd

from src.app.controllers import build_dashboard_state
from src.app.services.model_service import build_sample_predictions


class IntegrationArtifactSmokeTests(unittest.TestCase):
    def test_commit_message_impact_contains_jitline_rows(self) -> None:
        path = Path(__file__).resolve().parents[1] / "results" / "tables" / "commit_message_impact.csv"
        self.assertTrue(path.exists(), "commit_message_impact.csv must exist after running the impact script")
        df = pd.read_csv(path)
        self.assertFalse(df.empty)
        self.assertTrue({"openstack", "qt"}.issubset(set(df["dataset_name"].astype(str))))
        self.assertTrue(df["uses_commit_text_hybrid"].astype(bool).all())

    def test_hybrid_model_bundle_predicts_from_commit_text_sample(self) -> None:
        root = Path(__file__).resolve().parents[1]
        model_path = root / "models" / "hybrid_tfidf" / "xgb_openstack.joblib"
        data_path = root / "data" / "processed" / "openstack_clean.parquet"
        self.assertTrue(model_path.exists(), "xgb_openstack hybrid ModelBundle must exist")
        self.assertTrue(data_path.exists(), "openstack_clean.parquet must exist")

        bundle = joblib.load(model_path)
        sample = pd.read_parquet(data_path).head(3).copy()
        predictions, status = build_sample_predictions(
            {
                "model_path": str(model_path),
                "dataset_name": "openstack",
                "model": "xgb",
                "feature_family": "metrics_plus_commit_text",
                "uses_commit_text": True,
                "tfidf_num_features": 1000,
            },
            sample,
            [],
        )

        self.assertTrue(hasattr(bundle, "transform_features"))
        self.assertTrue(status.available, status.message)
        self.assertEqual(len(predictions), 3)
        self.assertIn("probability", predictions[0])

    def test_dashboard_resolves_final_hybrid_selection_for_openstack(self) -> None:
        state = build_dashboard_state("openstack")

        self.assertEqual(state.feature_family, "metrics_plus_commit_text")
        self.assertTrue(state.selected_model_row["uses_commit_text"])
        self.assertTrue(state.prediction_status.available, state.prediction_status.message)
        self.assertTrue(any(bool(row.get("uses_commit_text")) for row in state.ranking_rows))


if __name__ == "__main__":
    unittest.main()
