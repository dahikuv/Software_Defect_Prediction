"""Smoke tests for P3: GHPR adapter, commit-message impact, and hybrid bundle."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(script_name: str):
    script_path = PROTOTYPE_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CommitMessageImpactSmokeTests(unittest.TestCase):
    """Verify the commit-message impact table covers JITLine datasets when present."""

    def test_impact_table_contains_jitline_when_available(self) -> None:
        impact_path = PROTOTYPE_ROOT / "results" / "tables" / "commit_message_impact.csv"
        if not impact_path.exists():
            self.skipTest("commit_message_impact.csv not generated yet")

        df = pd.read_csv(impact_path)
        if df.empty:
            self.skipTest("impact table has no rows yet")

        self.assertIn("dataset_name", df.columns)
        self.assertIn("text_branch", df.columns)
        jitline = df[df["dataset_name"].astype(str).isin({"openstack", "qt", "jitfine"})]
        self.assertFalse(jitline.empty, "impact table should include openstack, qt, or jitfine rows once commit-level datasets are trained")
        for column in ("delta_f1", "delta_auc"):
            self.assertIn(column, df.columns)


class GhprAdapterSmokeTests(unittest.TestCase):
    """Verify GHPR adapter joins commit text by trimming the trailing label digit on SHA."""

    def test_prepare_ghpr_hybrid_frame_drops_conflicting_fix_sha_groups(self) -> None:
        module = _load_script_module("run_train_hybrid_tfidf.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            baseline_path = raw_dir / "baseline.csv"
            ghpr_path = raw_dir / "ghprdata.csv"
            conflict_sha = "a" * 40
            retained_sha = "b" * 40

            baseline_rows = [
                {"SHA": f"{conflict_sha}0", "defect": 0, "PROJECT_NAME": "GHPR"},
                {"SHA": f"{conflict_sha}1", "defect": 1, "PROJECT_NAME": "GHPR"},
                {"SHA": f"{retained_sha}0", "defect": 0, "PROJECT_NAME": "GHPR"},
            ]
            for row in baseline_rows:
                for metric in module.GHPR_METRIC_COLUMNS:
                    row[metric] = 1
            pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False)

            ghpr_rows = [
                [
                    "project",
                    "owner",
                    "description",
                    "label",
                    "python",
                    conflict_sha,
                    "bug",
                    "diff",
                    "fix conflict",
                    "time",
                    "old",
                    "new",
                    "old.py",
                    "new.py",
                    "title",
                    "body",
                ],
                [
                    "project",
                    "owner",
                    "description",
                    "label",
                    "python",
                    retained_sha,
                    "bug",
                    "diff",
                    "fix retained",
                    "time",
                    "old",
                    "new",
                    "old.py",
                    "new.py",
                    "title",
                    "body",
                ],
            ]
            pd.DataFrame(ghpr_rows).to_csv(ghpr_path, index=False, header=False)

            original_raw_dir = module.GHPR_RAW_DIR
            module.GHPR_RAW_DIR = raw_dir
            try:
                df = module.prepare_ghpr_hybrid_frame()
            finally:
                module.GHPR_RAW_DIR = original_raw_dir

        self.assertEqual(df.attrs.get("ghpr_pair_policy"), module.GHPR_PAIR_POLICY)
        self.assertEqual(df.attrs.get("ghpr_conflicting_fix_sha_count"), 1)
        self.assertEqual(df.attrs.get("ghpr_conflicting_rows_dropped"), 2)
        self.assertEqual(set(df["fix_sha"]), {retained_sha})
        self.assertFalse((df.groupby("fix_sha")["label"].nunique(dropna=False) > 1).any())

    def test_prepare_ghpr_hybrid_frame_attaches_commit_text(self) -> None:
        module = _load_script_module("run_train_hybrid_tfidf.py")
        if not module.GHPR_RAW_DIR.exists():
            self.skipTest("GHPR raw directory not present")
        if not (module.GHPR_RAW_DIR / "ghprdata.csv").exists():
            self.skipTest("GHPR ghprdata.csv not present")

        df = module.prepare_ghpr_hybrid_frame()
        self.assertIn("commit_text", df.columns)
        self.assertEqual(df.attrs.get("ghpr_pair_policy"), module.GHPR_PAIR_POLICY)
        if df.empty:
            self.assertGreater(df.attrs.get("ghpr_conflicting_fix_sha_count", 0), 0)
            return

        non_empty_ratio = df["commit_text"].astype(str).str.strip().ne("").mean()
        self.assertGreater(non_empty_ratio, 0.5, "GHPR adapter must populate commit_text for the majority of retained rows")
        label_counts_by_fix = df.groupby("fix_sha")["label"].nunique(dropna=False)
        self.assertFalse((label_counts_by_fix > 1).any(), "GHPR adapter must not retain conflicting labels for the same fix SHA")


class HybridBundlePredictionSmokeTests(unittest.TestCase):
    """Verify a saved hybrid ModelBundle can score a sample row that includes commit_text."""

    def test_jitline_hybrid_bundle_scores_sample_row(self) -> None:
        from src.app.services.model_service import build_sample_predictions

        bundle_path = PROTOTYPE_ROOT / "models" / "hybrid_tfidf" / "lgbm_openstack.joblib"
        clean_path = PROTOTYPE_ROOT / "data" / "processed" / "openstack_clean.parquet"
        if not bundle_path.exists() or not clean_path.exists():
            self.skipTest("hybrid bundle or processed dataset is not available")

        sample_df = pd.read_parquet(clean_path).head(3).copy()
        if sample_df.empty:
            self.skipTest("processed dataset is empty")

        if "module_id" not in sample_df.columns and "commit_id" in sample_df.columns:
            sample_df["module_id"] = sample_df["commit_id"].astype(str)

        selected_row = {
            "model_path": str(bundle_path),
            "dataset_name": "openstack",
            "model": "lgbm",
            "feature_family": "metrics_plus_commit_text",
            "uses_commit_text": True,
            "text_feature_column": "commit_text",
        }

        predictions, status = build_sample_predictions(selected_row, sample_df, ["la", "ld"])
        self.assertTrue(status.available, status.message)
        self.assertEqual(len(predictions), len(sample_df))
        for record in predictions:
            self.assertIn("prediction", record)
            self.assertIn("probability", record)


if __name__ == "__main__":
    unittest.main()
