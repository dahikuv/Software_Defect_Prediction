"""Run the full experiment pipeline end-to-end.

Steps (in order):
  1. run_data_pipeline        - ingest & clean raw datasets -> *_clean.parquet
  2. run_split_datasets        - create train/val/test ID files
  3. run_experiment_datasets   - build experiment-ready parquets for PROMISE
  4. run_ingest_jitline        - ingest JITLine (openstack, qt)
  5. run_ingest_jitfine        - ingest JIT-Fine (21 Apache projects)
  6. run_train_metrics_only    - metrics-only baseline (RF / XGB / LGBM)
  7. run_train_tuned_metrics   - CV-tuned metrics-only models
  8. run_train_hybrid_tfidf    - hybrid metrics + TF-IDF
  9. run_evaluation            - aggregate, rank, select final models
 10. run_commit_message_impact - metrics-only vs hybrid delta (paper §4.3)
 11. run_error_analysis        - TP/TN/FP/FN representative cases (§4.5)
 12. run_shap                  - SHAP global + local explainability (§4.4)

Usage:
    python scripts/run_all.py [--start STEP] [--stop STEP] [--skip STEP [STEP ...]]

    --start  1-based step number to start from (default: 1)
    --stop   1-based step number to stop after  (default: last)
    --skip   space-separated step numbers to skip

Examples:
    python scripts/run_all.py                     # full pipeline
    python scripts/run_all.py --start 7           # re-run from tuned training
    python scripts/run_all.py --skip 4 5          # skip JITLine / JITFine ingest
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _step(number: int, name: str, fn):
    return {"number": number, "name": name, "fn": fn}


def _build_steps() -> list[dict]:
    # Import lazily so missing optional deps only fail at the step that needs them.
    from scripts.run_data_pipeline import main as data_pipeline
    from scripts.run_split_datasets import main as split_datasets
    from scripts.run_experiment_datasets import main as experiment_datasets
    from scripts.run_ingest_jitline import main as ingest_jitline
    from scripts.run_ingest_jitfine import main as ingest_jitfine
    from scripts.run_train_metrics_only import main as train_metrics
    from scripts.run_train_tuned_metrics import main as train_tuned
    from scripts.run_train_hybrid_tfidf import main as train_hybrid
    from scripts.run_evaluation import main as evaluation
    from scripts.run_commit_message_impact import main as commit_impact
    from scripts.run_error_analysis import main as error_analysis
    from scripts.run_shap import main as shap

    return [
        _step(1,  "data_pipeline",         data_pipeline),
        _step(2,  "split_datasets",         split_datasets),
        _step(3,  "experiment_datasets",    experiment_datasets),
        _step(4,  "ingest_jitline",         ingest_jitline),
        _step(5,  "ingest_jitfine",         ingest_jitfine),
        _step(6,  "train_metrics_only",     train_metrics),
        _step(7,  "train_tuned_metrics",    train_tuned),
        _step(8,  "train_hybrid_tfidf",     train_hybrid),
        _step(9,  "evaluation",             evaluation),
        _step(10, "commit_message_impact",  commit_impact),
        _step(11, "error_analysis",         error_analysis),
        _step(12, "shap",                   shap),
    ]


def _parse_args(steps: list[dict]) -> argparse.Namespace:
    last = steps[-1]["number"]
    parser = argparse.ArgumentParser(description="Run the full defect-prediction pipeline.")
    parser.add_argument("--start", type=int, default=1, metavar="STEP",
                        help=f"First step to run (1–{last}, default: 1)")
    parser.add_argument("--stop", type=int, default=last, metavar="STEP",
                        help=f"Last step to run (1–{last}, default: {last})")
    parser.add_argument("--skip", type=int, nargs="+", default=[], metavar="STEP",
                        help="Step numbers to skip")
    return parser.parse_args()


def main() -> None:
    steps = _build_steps()
    args = _parse_args(steps)

    skip_set = set(args.skip)
    selected = [
        s for s in steps
        if args.start <= s["number"] <= args.stop and s["number"] not in skip_set
    ]

    if not selected:
        logger.warning("No steps selected. Check --start / --stop / --skip arguments.")
        return

    logger.info("=" * 60)
    logger.info("Pipeline: %s step(s) selected (start=%s stop=%s skip=%s)",
                len(selected), args.start, args.stop, sorted(skip_set) or "none")
    for s in selected:
        logger.info("  [%2d] %s", s["number"], s["name"])
    logger.info("=" * 60)

    failures: list[tuple[int, str, str]] = []
    total_start = time.perf_counter()

    for s in selected:
        num, name = s["number"], s["name"]
        logger.info("")
        logger.info(">>> STEP %d / %d  —  %s", num, steps[-1]["number"], name)
        t0 = time.perf_counter()
        try:
            s["fn"]()
            elapsed = time.perf_counter() - t0
            logger.info("<<< STEP %d done in %.1fs", num, elapsed)
        except Exception:
            elapsed = time.perf_counter() - t0
            tb = traceback.format_exc()
            logger.error("<<< STEP %d FAILED after %.1fs:\n%s", num, elapsed, tb)
            failures.append((num, name, tb))

    total_elapsed = time.perf_counter() - total_start
    logger.info("")
    logger.info("=" * 60)
    if failures:
        logger.error("Pipeline finished with %d failure(s) in %.1fs:", len(failures), total_elapsed)
        for num, name, _ in failures:
            logger.error("  [%2d] %s  — FAILED", num, name)
        sys.exit(1)
    else:
        logger.info("Pipeline completed successfully in %.1fs.", total_elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
