from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.app.services.evaluation_service import row_to_dict, select_model_row
from src.app.services.model_service import build_sample_predictions
from src.app.services.repo_analysis_service import (
    MAX_UPLOAD_BYTES,
    ProjectSource,
    _extract_github_owner_repo,
    _project_from_upload,
)
from src.evaluation.compare import select_final_models
from src.utils.provenance import artifact_uses_commit_text


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeProbabilityModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        scores = np.repeat(0.5, len(X))
        return np.column_stack([1.0 - scores, scores])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.repeat(1, len(X))


class UploadedFile:
    name = "sample.py"

    def getvalue(self) -> bytes:
        return b"def handler(request):\n    return request\n"


class OversizedUploadedFile:
    name = "large.py"

    def getvalue(self) -> bytes:
        return b"x" * (MAX_UPLOAD_BYTES + 1)


class ArtifactContractTests(unittest.TestCase):
    def test_artifact_uses_commit_text_never_infers_from_text_column_or_family(self) -> None:
        metadata = {
            "feature_family": "metrics_plus_commit_text",
            "text_feature_column": "commit_text",
        }

        self.assertFalse(artifact_uses_commit_text(metadata))
        self.assertTrue(artifact_uses_commit_text({**metadata, "uses_commit_text": True}))
        self.assertTrue(artifact_uses_commit_text({**metadata, "tfidf_num_features": 3}))

    def test_evaluation_row_normalization_does_not_infer_commit_text_from_text_column(self) -> None:
        row = pd.Series(
            {
                "dataset_name": "cm1",
                "model": "rf",
                "feature_family": "metrics_plus_commit_text",
                "text_feature_column": "commit_text",
            }
        )

        normalized = row_to_dict(row)

        self.assertFalse(normalized["uses_commit_text"])
        self.assertFalse(normalized["commit_text_available"])

    def test_evaluation_enrichment_uses_positive_feature_counts_for_commit_text(self) -> None:
        run_evaluation = _load_script_module("run_evaluation.py")
        source = Path("results.csv")
        df = pd.DataFrame(
            [
                {"dataset_name": "cm1", "model": "rf", "feature_family": "metrics_plus_commit_text", "text_feature_column": "commit_text"},
                {"dataset_name": "cm1", "model": "xgb", "feature_family": "metrics_plus_commit_text", "text_feature_column": "commit_text", "tfidf_num_features": 2},
            ]
        )

        enriched = run_evaluation._enrich_artifact_metadata(df, stage_name="test", source_results_table=source)

        self.assertFalse(bool(enriched.loc[0, "uses_commit_text"]))
        self.assertTrue(bool(enriched.loc[1, "uses_commit_text"]))

    def test_final_hybrid_dataset_resolver_uses_configured_exclusion(self) -> None:
        run_evaluation = _load_script_module("run_evaluation.py")
        config = {
            "features": {
                "hybrid": {
                    "datasets": ["ghpr", "openstack", "qt", "jitfine"],
                    "final_selection_excluded": ["ghpr"],
                }
            }
        }

        eligible = run_evaluation._resolve_final_hybrid_datasets(config)

        self.assertEqual(eligible, {"openstack", "qt", "jitfine"})

    def test_select_model_row_prefers_selected_ranked_commit_text_variant(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "dataset_name": "openstack",
                    "model": "lgbm",
                    "rank_within_dataset": 2,
                    "feature_family": "metrics_only",
                    "uses_commit_text": False,
                    "model_path": "baseline.joblib",
                },
                {
                    "dataset_name": "openstack",
                    "model": "lgbm",
                    "rank_within_dataset": 1,
                    "feature_family": "metrics_plus_commit_text",
                    "uses_commit_text": True,
                    "tfidf_num_features": 1000,
                    "model_path": "hybrid.joblib",
                },
            ]
        )

        selected = select_model_row(rows, "openstack", "lgbm")

        self.assertIsNotNone(selected)
        self.assertEqual(selected["feature_family"], "metrics_plus_commit_text")
        self.assertTrue(bool(selected["uses_commit_text"]))

    def test_split_primary_dataset_check_uses_config_provider(self) -> None:
        run_split = _load_script_module("run_split_datasets.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            primary = Path(tmpdir) / "configured.csv"
            other = Path(tmpdir) / "other.csv"
            primary.touch()
            other.touch()
            original_provider = run_split.primary_dataset_files_from_config
            run_split.primary_dataset_files_from_config = lambda: [primary]
            try:
                self.assertTrue(run_split.is_primary_dataset(primary))
                self.assertFalse(run_split.is_primary_dataset(other))
            finally:
                run_split.primary_dataset_files_from_config = original_provider

    def test_feature_pipeline_primary_dataset_check_uses_config_provider(self) -> None:
        run_feature = _load_script_module("run_feature_pipeline.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            primary = Path(tmpdir) / "configured.csv"
            other = Path(tmpdir) / "legacy.csv"
            primary.touch()
            other.touch()
            original_provider = run_feature.primary_dataset_files_from_config
            run_feature.primary_dataset_files_from_config = lambda: [primary]
            try:
                self.assertTrue(run_feature.is_primary_dataset(primary))
                self.assertFalse(run_feature.is_primary_dataset(other))
            finally:
                run_feature.primary_dataset_files_from_config = original_provider

    def test_experiment_frame_keeps_raw_metric_missing_values(self) -> None:
        run_experiment = _load_script_module("run_experiment_datasets.py")
        df = pd.DataFrame(
            {
                "module_id": ["m1", "m2", "m3"],
                "label": [0, 1, 0],
                "loc": [10.0, None, 30.0],
                "v(g)": [1.0, 2.0, 3.0],
            }
        )

        experiment_df, metadata = run_experiment.build_experiment_frame(df, ["loc", "v(g)", "missing_metric"])

        self.assertTrue(pd.isna(experiment_df.loc[1, "loc"]))
        self.assertEqual(metadata["imputation_fit_scope"], "train_split_only")
        self.assertEqual(metadata["selected_metrics"], ["loc", "v(g)"])
        self.assertEqual(metadata["missing_metrics"], ["missing_metric"])

    def test_tuned_candidate_selection_ignores_test_metrics(self) -> None:
        run_tuned = _load_script_module("run_train_tuned_metrics.py")
        selected = run_tuned.select_best_candidate(
            [
                {
                    "candidate_index": 1,
                    "threshold_constraint_met": True,
                    "cv_mean_recall": 0.8,
                    "cv_mean_f1": 0.7,
                    "cv_mean_auc": 0.75,
                    "val_recall": 0.8,
                    "val_f1": 0.7,
                    "val_auc": 0.75,
                    "val_precision": 0.6,
                    "gap_train_val_f1": 0.1,
                    "gap_train_val_auc": 0.1,
                    "recall": 0.1,
                    "f1": 0.1,
                    "precision": 0.1,
                },
                {
                    "candidate_index": 2,
                    "threshold_constraint_met": True,
                    "cv_mean_recall": 0.4,
                    "cv_mean_f1": 0.4,
                    "cv_mean_auc": 0.4,
                    "val_recall": 0.4,
                    "val_f1": 0.4,
                    "val_auc": 0.4,
                    "val_precision": 0.4,
                    "gap_train_val_f1": 0.0,
                    "gap_train_val_auc": 0.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "precision": 1.0,
                },
            ]
        )

        self.assertEqual(selected["candidate_index"], 1)

    def test_final_model_selection_prefers_validation_over_test_metrics(self) -> None:
        baseline = pd.DataFrame(
            [
                {
                    "dataset_name": "cm1",
                    "model": "rf",
                    "val_recall": 0.9,
                    "val_f1": 0.8,
                    "val_auc": 0.8,
                    "val_precision": 0.7,
                    "recall": 0.1,
                    "f1": 0.1,
                    "auc": 0.1,
                    "precision": 0.1,
                    "threshold_constraint_met": True,
                }
            ]
        )
        tuned = pd.DataFrame(
            [
                {
                    "dataset_name": "cm1",
                    "model": "xgb",
                    "val_recall": 0.2,
                    "val_f1": 0.2,
                    "val_auc": 0.2,
                    "val_precision": 0.2,
                    "recall": 1.0,
                    "f1": 1.0,
                    "auc": 1.0,
                    "precision": 1.0,
                    "threshold_constraint_met": True,
                }
            ]
        )

        selected = select_final_models(baseline, tuned, selection_policy="best_validation")

        self.assertEqual(selected.iloc[0]["model"], "rf")
        self.assertTrue(bool(selected.iloc[0]["test_metrics_report_only"]))
        self.assertEqual(selected.iloc[0]["selection_data_source"], "validation")

    def test_final_model_selection_default_policy_prefers_tuned_within_dataset(self) -> None:
        baseline = pd.DataFrame(
            [
                {
                    "dataset_name": "cm1",
                    "model": "rf",
                    "val_recall": 0.9,
                    "val_f1": 0.8,
                    "val_auc": 0.8,
                    "val_precision": 0.7,
                    "threshold_constraint_met": True,
                }
            ]
        )
        tuned = pd.DataFrame(
            [
                {
                    "dataset_name": "cm1",
                    "model": "xgb",
                    "val_recall": 0.2,
                    "val_f1": 0.2,
                    "val_auc": 0.2,
                    "val_precision": 0.2,
                    "threshold_constraint_met": True,
                }
            ]
        )

        selected = select_final_models(baseline, tuned)

        self.assertEqual(selected.iloc[0]["model"], "xgb")
        self.assertEqual(selected.iloc[0]["training_mode"], "tuned")
        self.assertEqual(selected.iloc[0]["selection_policy"], "tuned_first")

    def test_final_model_selection_hybrid_policy_selects_valid_hybrid_when_validation_better(self) -> None:
        baseline = pd.DataFrame(
            [
                {
                    "dataset_name": "openstack",
                    "model": "rf",
                    "val_recall": 0.4,
                    "val_f1": 0.3,
                    "threshold_constraint_met": True,
                }
            ]
        )
        tuned = pd.DataFrame(
            [
                {
                    "dataset_name": "openstack",
                    "model": "xgb",
                    "val_recall": 0.6,
                    "val_f1": 0.5,
                    "threshold_constraint_met": True,
                }
            ]
        )
        hybrid = pd.DataFrame(
            [
                {
                    "dataset_name": "openstack",
                    "model": "lgbm",
                    "feature_family": "metrics_plus_commit_text",
                    "uses_commit_text": True,
                    "tfidf_num_features": 100,
                    "val_recall": 0.7,
                    "val_f1": 0.55,
                    "threshold_constraint_met": True,
                }
            ]
        )

        selected = select_final_models(
            baseline_best_df=baseline,
            tuned_best_df=tuned,
            hybrid_best_df=hybrid,
            selection_policy="hybrid_validation_then_tuned",
        )

        self.assertEqual(selected.iloc[0]["model"], "lgbm")
        self.assertEqual(selected.iloc[0]["training_mode"], "hybrid_tfidf")
        self.assertEqual(selected.iloc[0]["selection_policy"], "hybrid_validation_then_tuned")

    def test_final_model_selection_hybrid_policy_ignores_invalid_hybrid_rows(self) -> None:
        baseline = pd.DataFrame(
            [
                {
                    "dataset_name": "openstack",
                    "model": "rf",
                    "val_recall": 0.4,
                    "val_f1": 0.3,
                    "threshold_constraint_met": True,
                }
            ]
        )
        tuned = pd.DataFrame(
            [
                {
                    "dataset_name": "openstack",
                    "model": "xgb",
                    "val_recall": 0.6,
                    "val_f1": 0.5,
                    "threshold_constraint_met": True,
                }
            ]
        )
        hybrid = pd.DataFrame(
            [
                {
                    "dataset_name": "openstack",
                    "model": "lgbm",
                    "feature_family": "metrics_plus_commit_text",
                    "uses_commit_text": False,
                    "tfidf_num_features": 0,
                    "val_recall": 0.9,
                    "val_f1": 0.8,
                    "threshold_constraint_met": True,
                }
            ]
        )

        selected = select_final_models(
            baseline_best_df=baseline,
            tuned_best_df=tuned,
            hybrid_best_df=hybrid,
            selection_policy="hybrid_validation_then_tuned",
        )

        self.assertEqual(selected.iloc[0]["model"], "xgb")
        self.assertEqual(selected.iloc[0]["training_mode"], "tuned")

    def test_final_model_selection_tuned_first_falls_back_to_baseline_when_tuned_missing(self) -> None:
        baseline = pd.DataFrame(
            [
                {
                    "dataset_name": "cm1",
                    "model": "rf",
                    "val_recall": 0.9,
                    "val_f1": 0.8,
                    "threshold_constraint_met": True,
                },
                {
                    "dataset_name": "kc1",
                    "model": "rf",
                    "val_recall": 0.5,
                    "val_f1": 0.4,
                    "threshold_constraint_met": True,
                },
            ]
        )
        tuned = pd.DataFrame(
            [
                {
                    "dataset_name": "kc1",
                    "model": "xgb",
                    "val_recall": 0.7,
                    "val_f1": 0.6,
                    "threshold_constraint_met": True,
                }
            ]
        )

        selected = select_final_models(baseline, tuned).sort_values("dataset_name").reset_index(drop=True)

        self.assertEqual(list(selected["dataset_name"]), ["cm1", "kc1"])
        self.assertEqual(selected.loc[0, "model"], "rf")
        self.assertEqual(selected.loc[0, "training_mode"], "baseline")
        self.assertEqual(selected.loc[1, "model"], "xgb")
        self.assertEqual(selected.loc[1, "training_mode"], "tuned")

    def test_final_model_selection_rejects_unknown_policy(self) -> None:
        baseline = pd.DataFrame([{"dataset_name": "cm1", "model": "rf", "val_recall": 0.5, "threshold_constraint_met": True}])
        tuned = pd.DataFrame([{"dataset_name": "cm1", "model": "xgb", "val_recall": 0.6, "threshold_constraint_met": True}])
        with self.assertRaises(ValueError):
            select_final_models(baseline, tuned, selection_policy="bogus")

    def test_upload_project_source_dataclass_is_available(self) -> None:
        project, excluded = _project_from_upload(UploadedFile())

        self.assertIsInstance(project, ProjectSource)
        self.assertEqual(excluded, [])
        self.assertEqual(project.source_type, "upload")
        self.assertEqual(len(project.snapshots), 1)

    def test_upload_project_source_rejects_oversized_files(self) -> None:
        project, excluded = _project_from_upload(OversizedUploadedFile())

        self.assertEqual(excluded, [])
        self.assertEqual(project.snapshots, [])
        self.assertIn("too large", project.notes[0])

    def test_repo_url_parser_accepts_only_safe_https_github_urls(self) -> None:
        self.assertEqual(_extract_github_owner_repo("https://github.com/example/repo.git"), ("example", "repo"))
        self.assertIsNone(_extract_github_owner_repo("http://github.com/example/repo"))
        self.assertIsNone(_extract_github_owner_repo("https://github.com.evil/example/repo"))
        self.assertIsNone(_extract_github_owner_repo("https://github.com/example/bad repo"))

    def test_legacy_hybrid_model_prediction_skips_instead_of_refitting_text_features(self) -> None:
        sample_df = pd.DataFrame(
            {
                "module_id": ["m1"],
                "label": [0],
                "loc": [10],
                "commit_text": ["fix null handling"],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "legacy.joblib"
            joblib.dump(FakeProbabilityModel(), model_path)
            predictions, status = build_sample_predictions(
                {
                    "model_path": str(model_path),
                    "dataset_name": "cm1",
                    "model": "rf",
                    "feature_family": "metrics_plus_commit_text",
                    "uses_commit_text": True,
                },
                sample_df,
                ["loc"],
            )

        self.assertEqual(predictions, [])
        self.assertFalse(status.available)
        self.assertTrue(status.details["requires_model_bundle"])
        self.assertTrue(status.details["commit_text_inference_skipped"])


if __name__ == "__main__":
    unittest.main()
