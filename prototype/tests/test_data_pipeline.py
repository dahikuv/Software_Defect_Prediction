from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.clean import clean_dataset
from src.data.ingest import build_dataset_inventory, classify_dataset, load_dataset, prepare_dataset_from_raw
from src.data.split import stratified_split
from src.data.unify_schema import unify_schema
from src.data.validate import ensure_non_empty_columns, validate_dataset_schema, validate_required_columns


class IngestTests(unittest.TestCase):
    def test_classify_dataset_uses_config_tiers(self) -> None:
        tier = classify_dataset("nonexistent_dataset")
        self.assertIn(tier, {"unknown", "candidate_primary", "primary", "supplementary", "excluded"})

    def test_build_dataset_inventory_returns_dataframe(self) -> None:
        inventory = build_dataset_inventory(r"C:/Users/Dam Hieu/Desktop/CNDM - VIẾT BÁO + KLTN/prototype/data/raw")
        self.assertIsInstance(inventory, pd.DataFrame)

    def test_prepare_dataset_from_raw_processes_real_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            pd.DataFrame(
                {
                    "file": ["m1", "m2"],
                    "project": ["proj", "proj"],
                    "commit_message": ["Fix bug", "Refactor"],
                    "defect": ["yes", "no"],
                    "loc": [10, 20],
                }
            ).to_csv(path, index=False)

            cleaned_df, profile = prepare_dataset_from_raw(path)

            self.assertIn("label", cleaned_df.columns)
            self.assertIn("module_id", cleaned_df.columns)
            self.assertEqual(profile["dataset_name"], "sample")
            self.assertEqual(profile["status"], "ok")
            self.assertIn("clean_summary", profile)
            self.assertEqual(len(cleaned_df), 2)

    def test_prepare_dataset_from_raw_loads_callable(self) -> None:
        self.assertTrue(callable(load_dataset))
        self.assertTrue(callable(prepare_dataset_from_raw))


class DataPipelineTests(unittest.TestCase):
    def test_validate_required_columns_raises_when_missing(self) -> None:
        df = pd.DataFrame({"module_id": ["a"]})
        with self.assertRaises(ValueError):
            validate_required_columns(df)

    def test_validate_dataset_schema_accepts_binary_labels(self) -> None:
        df = pd.DataFrame({"module_id": ["a", "b"], "label": [0, 1]})
        validate_dataset_schema(df)

    def test_validate_dataset_schema_rejects_empty_frame(self) -> None:
        df = pd.DataFrame({"module_id": [], "label": []})
        with self.assertRaises(ValueError):
            validate_dataset_schema(df)

    def test_unify_schema_normalizes_aliases_and_labels(self) -> None:
        df = pd.DataFrame(
            {
                "file": ["mod1", "mod2"],
                "project": ["proj", "proj"],
                "commit_message": ["Fix bug", "Refactor"],
                "defect": ["yes", "no"],
            }
        )
        unified = unify_schema(df, dataset_name="sample.csv")
        self.assertListEqual(list(unified["module_id"]), ["mod1", "mod2"])
        self.assertListEqual(list(unified["project_name"]), ["proj", "proj"])
        self.assertListEqual(list(unified["label"]), [1, 0])
        self.assertTrue((unified["commit_text"] == ["Fix bug", "Refactor"]).all())

    def test_clean_dataset_returns_summary_and_imputes_numeric(self) -> None:
        df = pd.DataFrame(
            {
                "module_id": ["a", "a", "b"],
                "project_name": [" p ", " p ", "q"],
                "commit_text": [" fix ", " fix ", None],
                "label": [1, 1, 0],
                "loc": [10.0, 10.0, None],
            }
        )
        cleaned, summary = clean_dataset(df, deduplicate_by_module_id=True, return_summary=True)
        self.assertEqual(summary["rows_before"], 3)
        self.assertGreaterEqual(summary["exact_duplicates_removed"], 1)
        self.assertEqual(summary["rows_after"], len(cleaned))
        self.assertTrue(cleaned["loc"].isna().sum() == 0)

    def test_stratified_split_validates_and_splits(self) -> None:
        df = pd.DataFrame(
            {
                "module_id": [f"m{i}" for i in range(10)],
                "label": [0, 1] * 5,
                "loc": list(range(10)),
            }
        )
        train_df, val_df, test_df = stratified_split(df, test_size=0.2, val_size=0.2, random_state=1)
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(df))
        self.assertIn("label", train_df.columns)

    def test_stratified_split_rejects_missing_label(self) -> None:
        df = pd.DataFrame({"module_id": ["a", "b"]})
        with self.assertRaises(ValueError):
            stratified_split(df)

    def test_ensure_non_empty_columns_raises_for_all_nan_column(self) -> None:
        df = pd.DataFrame({"module_id": [None, None], "label": [0, 1]})
        with self.assertRaises(ValueError):
            ensure_non_empty_columns(df, ["module_id"])


if __name__ == "__main__":
    unittest.main()
