from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.jitfine_ingest import (
    JITFINE_DATASET_NAME,
    JITFINE_METRIC_COLUMNS,
    build_jitfine_dataset,
    jitfine_artifacts_available,
    resolve_jitfine_feature_paths,
)


def _build_raw_frame(commit_ids, projects, messages, labels, metric_overrides=None):
    metric_overrides = metric_overrides or {}
    rows = []
    for commit_id, project, message, label in zip(commit_ids, projects, messages, labels):
        row = {
            'commit_hash': commit_id,
            'project': project,
            'commit_message': message,
            'is_buggy_commit': label,
            'fileschanged': '[]',
            'classification': 'unknown',
            'fix': 0,
        }
        for column in JITFINE_METRIC_COLUMNS:
            row[column] = metric_overrides.get((commit_id, column), 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.write_bytes(pickle.dumps(frame))


class JitfineIngestTests(unittest.TestCase):
    def test_resolve_jitfine_feature_paths_uses_canonical_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            paths = resolve_jitfine_feature_paths(raw_dir=base)
            self.assertEqual(paths['train'], base / 'JITFine' / 'features_train.pkl')
            self.assertEqual(paths['val'], base / 'JITFine' / 'features_valid.pkl')
            self.assertEqual(paths['test'], base / 'JITFine' / 'features_test.pkl')

    def test_jitfine_artifacts_available_requires_three_pickles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / 'JITFine'
            base.mkdir(parents=True)
            self.assertFalse(jitfine_artifacts_available(raw_dir=Path(tmpdir)))
            for filename in ('features_train.pkl', 'features_valid.pkl', 'features_test.pkl'):
                (base / filename).write_bytes(b'\x80\x04N.')
            self.assertTrue(jitfine_artifacts_available(raw_dir=Path(tmpdir)))

    def test_build_jitfine_dataset_joins_three_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / 'JITFine'
            base.mkdir(parents=True)
            train_frame = _build_raw_frame(
                ['abc'],
                ['hadoop'],
                ['fix null'],
                [1],
                {('abc', 'la'): 12.0, ('abc', 'ld'): 3.0},
            )
            val_frame = _build_raw_frame(['def'], ['cassandra'], ['refactor'], [0])
            test_frame = _build_raw_frame(['ghi'], ['ignite'], ['add docs'], [0])
            _write_frame(base / 'features_train.pkl', train_frame)
            _write_frame(base / 'features_valid.pkl', val_frame)
            _write_frame(base / 'features_test.pkl', test_frame)

            dataset = build_jitfine_dataset(raw_dir=Path(tmpdir))

            self.assertSetEqual(set(dataset['module_id'].tolist()), {'abc', 'def', 'ghi'})
            self.assertSetEqual(set(dataset['jitline_split'].tolist()), {'train', 'val', 'test'})
            self.assertEqual(dataset['dataset_name'].iloc[0], JITFINE_DATASET_NAME)
            self.assertEqual(int(dataset.loc[dataset['module_id'] == 'abc', 'la'].iloc[0]), 12)
            self.assertEqual(dataset.loc[dataset['module_id'] == 'abc', 'commit_text'].iloc[0], 'fix null')
            self.assertEqual(int(dataset.loc[dataset['module_id'] == 'abc', 'label'].iloc[0]), 1)
            self.assertNotIn('fileschanged', dataset.columns)
            self.assertNotIn('classification', dataset.columns)


if __name__ == '__main__':
    unittest.main()