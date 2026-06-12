from __future__ import annotations

import unittest

import pandas as pd

from src.features.commit_sbert import build_sbert_features
from src.features.commit_tfidf import build_tfidf_features, normalize_commit_text
from src.features.feature_merge import merge_feature_sets
from src.features.metrics_features import build_metrics_features, build_metrics_training_frame, get_available_metrics, summarize_metric_coverage


class MetricsFeatureTests(unittest.TestCase):
    def test_get_available_metrics(self) -> None:
        df = pd.DataFrame({"loc": [1, 2], "ev(g)": [3, 4]})
        available, missing = get_available_metrics(df, ["loc", "v(g)", "ev(g)"])
        self.assertEqual(available, ["loc", "ev(g)"])
        self.assertEqual(missing, ["v(g)"])

    def test_build_metrics_features_coerces_and_fills(self) -> None:
        df = pd.DataFrame({"loc": [1, None, 3], "ev(g)": ["2", "x", "4"], "label": [0, 1, 0]})
        features, metadata = build_metrics_features(df, ["loc", "ev(g)"], return_metadata=True)
        self.assertEqual(list(features.columns), ["loc", "ev(g)"])
        self.assertEqual(metadata["num_features"], 2)
        self.assertFalse(features.isna().any().any())

    def test_build_metrics_training_frame(self) -> None:
        df = pd.DataFrame({"loc": [1, 2], "ev(g)": [3, 4], "label": [0, 1]})
        X, y, metadata = build_metrics_training_frame(df, ["loc", "ev(g)"])
        self.assertEqual(len(X), 2)
        self.assertListEqual(y.tolist(), [0, 1])
        self.assertIn("label_distribution", metadata)

    def test_summarize_metric_coverage(self) -> None:
        df = pd.DataFrame({"loc": [1], "ev(g)": [2]})
        summary = summarize_metric_coverage(df, ["loc", "v(g)", "ev(g)"])
        self.assertAlmostEqual(summary["coverage_ratio"], 2 / 3)


class CommitFeatureTests(unittest.TestCase):
    def test_normalize_commit_text(self) -> None:
        series = pd.Series([" Fix Bug ", None])
        normalized = normalize_commit_text(series)
        self.assertListEqual(normalized.tolist(), ["fix bug", ""])

    def test_build_tfidf_features(self) -> None:
        series = pd.Series(["fix bug", "refactor code"])
        vectorizer, features = build_tfidf_features(series, max_features=10, ngram_range=(1, 1))
        self.assertGreaterEqual(features.shape[1], 1)
        self.assertEqual(features.shape[0], 2)
        self.assertTrue(hasattr(vectorizer, "get_feature_names_out"))

    def test_build_sbert_features_returns_dataframe(self) -> None:
        series = pd.Series(["fix bug", "refactor code"])
        features, metadata = build_sbert_features(series, model_name="all-MiniLM-L6-v2", return_metadata=True)
        self.assertEqual(len(features), 2)
        self.assertIn("model_name", metadata)
        self.assertIn("used_fallback", metadata)
        self.assertEqual(features.index.tolist(), series.index.tolist())


class FeatureMergeTests(unittest.TestCase):
    def test_merge_feature_sets(self) -> None:
        base_df = pd.DataFrame({"module_id": ["m1", "m2"]})
        metrics_df = pd.DataFrame({"loc": [1, 2]})
        text_df = pd.DataFrame({"tfidf_a": [0.1, 0.2]})
        merged = merge_feature_sets(base_df, metrics_df, text_df)
        self.assertListEqual(list(merged.columns), ["module_id", "metric_loc", "commit_tfidf_a"])
        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()
