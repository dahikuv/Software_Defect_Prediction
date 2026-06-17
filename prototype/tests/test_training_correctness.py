from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]

def _load_script_module(script_name: str):
    script_path = PROTOTYPE_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

class TunedSelectionCorrectnessTests(unittest.TestCase):
    def test_best_candidate_prefers_validation_threshold_metrics_over_raw_cv(self) -> None:
        module = _load_script_module("run_train_tuned_metrics.py")
        candidates = [
            {
                "candidate_index": 1,
                "threshold_constraint_met": True,
                "val_recall": 0.40,
                "val_f1": 0.45,
                "val_auc": 0.60,
                "val_precision": 0.50,
                "cv_mean_recall": 0.95,
                "cv_mean_f1": 0.90,
                "cv_mean_auc": 0.90,
                "gap_train_val_f1": 0.10,
                "gap_train_val_auc": 0.10,
            },
            {
                "candidate_index": 2,
                "threshold_constraint_met": True,
                "val_recall": 0.80,
                "val_f1": 0.70,
                "val_auc": 0.65,
                "val_precision": 0.55,
                "cv_mean_recall": 0.50,
                "cv_mean_f1": 0.50,
                "cv_mean_auc": 0.50,
                "gap_train_val_f1": 0.10,
                "gap_train_val_auc": 0.10,
            },
        ]

        selected = module.select_best_candidate(candidates)

        self.assertEqual(selected["candidate_index"], 2)

if __name__ == "__main__":
    unittest.main()
