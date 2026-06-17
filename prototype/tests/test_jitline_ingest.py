from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.jitline_ingest import (
    JITLINE_METRIC_COLUMNS,
    available_jitline_projects,
    build_jitline_dataset,
    jitline_raw_dir,
    load_jitline_metrics,
    resolve_jitline_paths,
)

def _write_text_pickle(path: Path, ids: list[str], labels: list[int], messages: list[str]) -> None:
    bundle = (
        list(ids),
        list(labels),
        list(messages),
        [["added _ code removed _ code"] for _ in ids],
    )
    path.write_bytes(pickle.dumps(bundle))

class JitlineIngestTests(unittest.TestCase):
    def test_resolve_jitline_paths_uses_canonical_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            paths = resolve_jitline_paths("openstack", raw_dir=base)
            self.assertEqual(paths.metrics_csv, base / "JITLine" / "openstack_metrics.csv")
            self.assertEqual(paths.train_pkl, base / "JITLine" / "openstack_train.pkl")
            self.assertEqual(paths.test_pkl, base / "JITLine" / "openstack_test.pkl")

    def test_resolve_jitline_paths_rejects_unknown_project(self) -> None:
        with self.assertRaises(ValueError):
            resolve_jitline_paths("postgres")

    def test_load_jitline_metrics_keeps_only_known_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "JITLine"
            base.mkdir(parents=True)
            metrics_path = base / "openstack_metrics.csv"
            pd.DataFrame(
                [
                    {"commit_id": "a", "la": 1, "ld": 2, "ent": 0.5, "bugcount": 9, "extra": "x"},
                    {"commit_id": "b", "la": 3, "ld": 4, "ent": 0.7, "bugcount": 1, "extra": "y"},
                ]
            ).to_csv(metrics_path, index=False)
            paths = resolve_jitline_paths("openstack", raw_dir=Path(tmpdir))
            paths_train = base / "openstack_train.pkl"
            paths_test = base / "openstack_test.pkl"
            paths_train.write_bytes(b"")
            paths_test.write_bytes(b"")

            metrics = load_jitline_metrics(paths, metric_columns=("la", "ld", "ent", "missing"))

            self.assertListEqual(list(metrics.columns), ["commit_id", "la", "ld", "ent"])
            self.assertNotIn("bugcount", metrics.columns)

    def test_build_jitline_dataset_joins_text_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "JITLine"
            base.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"commit_id": "abc", "la": 5, "ld": 1, "ent": 0.1},
                    {"commit_id": "def", "la": 0, "ld": 0, "ent": 0.0},
                ]
            ).to_csv(base / "openstack_metrics.csv", index=False)
            _write_text_pickle(base / "openstack_train.pkl", ["abc"], [1], ["fix null pointer"])
            _write_text_pickle(base / "openstack_test.pkl", ["def"], [0], ["docs update"])

            dataset = build_jitline_dataset("openstack", raw_dir=Path(tmpdir))

            self.assertListEqual(sorted(dataset["commit_id"].tolist()), ["abc", "def"])
            self.assertSetEqual(set(dataset["jitline_split"].tolist()), {"train", "test"})
            self.assertIn("la", dataset.columns)
            self.assertIn("commit_text", dataset.columns)
            train_row = dataset.loc[dataset["commit_id"] == "abc"].iloc[0]
            self.assertEqual(train_row["label"], 1)
            self.assertEqual(train_row["commit_text"], "fix null pointer")

    def test_available_jitline_projects_filters_to_complete_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "JITLine"
            base.mkdir(parents=True)
            (base / "openstack_metrics.csv").write_text("commit_id\nx\n", encoding="utf-8")
            _write_text_pickle(base / "openstack_train.pkl", ["x"], [0], ["msg"])
            _write_text_pickle(base / "openstack_test.pkl", ["x"], [0], ["msg"])
            (base / "qt_metrics.csv").write_text("commit_id\ny\n", encoding="utf-8")
            self.assertEqual(available_jitline_projects(raw_dir=Path(tmpdir)), ["openstack"])

    def test_jitline_metric_columns_excludes_known_leaky_metrics(self) -> None:
        for leaky in ("bugcount", "fixcount", "revd", "tcmt", "oexp", "orexp", "osexp", "osawr"):
            self.assertNotIn(leaky, JITLINE_METRIC_COLUMNS)

if __name__ == "__main__":
    unittest.main()
