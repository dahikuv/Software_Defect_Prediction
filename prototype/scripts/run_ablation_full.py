"""
Run full ablation study: Metrics-only vs TF-IDF-only vs Hybrid
for the 3 JIT datasets, reporting all 5 metrics (Accuracy, Precision, Recall, F1, AUC).

Outputs:
  - results/tables/ablation_full.csv           (per model)
  - results/tables/ablation_full_summary.csv   (mean across models per dataset)
  - results/tables/ablation_full_improvement.csv (% improvement vs metrics-only baseline)
"""

import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results" / "tables"
JIT_DATASETS = ["jitfine", "openstack", "qt"]
MODELS = ["rf", "xgb", "lgbm"]
METRICS = ["accuracy", "precision", "recall", "f1", "auc"]

# --- Load data ---
tfidf = pd.read_csv(RESULTS / "ablation_tfidf_only.csv")
hybrid = pd.read_csv(RESULTS / "hybrid_tfidf_results.csv")
tuned = pd.read_csv(RESULTS / "baseline_vs_tuned_comparison.csv")

# Filter: JIT datasets only, tuned rows (best config per model)
tuned_jit = tuned[
    (tuned["dataset_name"].isin(JIT_DATASETS))
    & (tuned["artifact_stage"] == "tuned_best_table")
].copy()

# --- Build unified table ---
rows = []
for ds in JIT_DATASETS:
    for m in MODELS:
        # Metrics-only (tuned)
        row_m = tuned_jit[(tuned_jit["dataset_name"] == ds) & (tuned_jit["model"] == m)]
        if row_m.empty:
            continue
        row_m = row_m.iloc[0]
        rows.append({
            "dataset": ds, "model": m.upper(), "feature_family": "Metrics-only",
            "accuracy": row_m["tuned_accuracy"], "precision": row_m["tuned_precision"],
            "recall": row_m["tuned_recall"], "f1": row_m["tuned_f1"], "auc": row_m["tuned_auc"],
        })

        # TF-IDF-only
        row_t = tfidf[(tfidf["dataset_name"] == ds) & (tfidf["model"] == m)]
        if not row_t.empty:
            row_t = row_t.iloc[0]
            rows.append({
                "dataset": ds, "model": m.upper(), "feature_family": "TF-IDF-only",
                "accuracy": row_t["accuracy"], "precision": row_t["precision"],
                "recall": row_t["recall"], "f1": row_t["f1"], "auc": row_t["auc"],
            })

        # Hybrid
        row_h = hybrid[(hybrid["dataset_name"] == ds) & (hybrid["model"] == m)]
        if not row_h.empty:
            row_h = row_h.iloc[0]
            rows.append({
                "dataset": ds, "model": m.upper(), "feature_family": "Hybrid",
                "accuracy": row_h["accuracy"], "precision": row_h["precision"],
                "recall": row_h["recall"], "f1": row_h["f1"], "auc": row_h["auc"],
            })

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "ablation_full.csv", index=False)
print("=== Ablation Full (per model) ===")
print(df.to_string(index=False, float_format="%.3f"))

# --- Summary: mean across models per dataset ---
summary = df.groupby(["dataset", "feature_family"])[METRICS].mean().reset_index()
summary.to_csv(RESULTS / "ablation_full_summary.csv", index=False)
print("\n=== Summary (mean across models) ===")
print(summary.to_string(index=False, float_format="%.3f"))

# --- Improvement table ---
improvements = []
for ds in JIT_DATASETS:
    for mf in METRICS:
        baseline_vals = df[(df["dataset"] == ds) & (df["feature_family"] == "Metrics-only")][mf]
        hybrid_vals = df[(df["dataset"] == ds) & (df["feature_family"] == "Hybrid")][mf]
        tfidf_vals = df[(df["dataset"] == ds) & (df["feature_family"] == "TF-IDF-only")][mf]

        if baseline_vals.empty or hybrid_vals.empty:
            continue

        b_mean = baseline_vals.mean()
        h_mean = hybrid_vals.mean()
        t_mean = tfidf_vals.mean() if not tfidf_vals.empty else float("nan")

        improvements.append({
            "dataset": ds, "metric": mf,
            "metrics_only_mean": b_mean,
            "tfidf_only_mean": t_mean,
            "hybrid_mean": h_mean,
            "improvement_metrics_to_hybrid": h_mean - b_mean,
            "improvement_pct_metrics_to_hybrid": ((h_mean - b_mean) / b_mean * 100) if b_mean != 0 else float("nan"),
            "improvement_tfidf_to_hybrid": h_mean - t_mean,
        })

imp_df = pd.DataFrame(improvements)
imp_df.to_csv(RESULTS / "ablation_full_improvement.csv", index=False)
print("\n=== Improvement (mean across models) ===")
print(imp_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
print(f"\nSaved: {RESULTS / 'ablation_full.csv'}")
print(f"Saved: {RESULTS / 'ablation_full_summary.csv'}")
print(f"Saved: {RESULTS / 'ablation_full_improvement.csv'}")
