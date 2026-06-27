"""
Generate a consolidated statistical significance test table for the paper.

Output: results/tables/statistical_tests_summary.csv
"""

import pandas as pd
import sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

RESULTS = Path(__file__).resolve().parent.parent / "results" / "tables"

df = pd.read_csv(RESULTS / "evaluation_significance.csv")

# Filter: only baseline_vs_hybrid comparisons
comp = df[df["comparison"] == "baseline_vs_hybrid_tfidf"].copy()

rows = []
for _, r in comp.iterrows():
    ds = r["dataset_name"]
    metric = r["metric"]
    delta = r["delta"]

    # Determine p-value and test type
    if pd.notna(r.get("delong_p_value_bh")) and metric == "auc":
        p_val = r["delong_p_value_bh"]
        test = "DeLong"
        sig = r.get("delong_significant_bh", False)
    elif pd.notna(r.get("permutation_p_value_bh")) and metric == "recall":
        p_val = r["permutation_p_value_bh"]
        test = "Permutation (300 iter)"
        sig = r.get("permutation_significant_bh", False)
    else:
        p_val = r.get("permutation_p_value_bh", float("nan"))
        test = "Permutation (300 iter)"
        sig = r.get("permutation_significant_bh", False)

    rows.append({
        "dataset": ds.upper(),
        "metric": metric.upper(),
        "delta": delta,
        "statistical_test": test,
        "p_value_bh": p_val,
        "significant_alpha005": "Yes" if sig else "No",
    })

out = pd.DataFrame(rows)
out.to_csv(RESULTS / "statistical_tests_summary.csv", index=False)

print("=== Statistical Tests Summary ===")
print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Count significant
n_sig = out[out["significant_alpha005"] == "Yes"].shape[0]
n_total = out.shape[0]
print(f"\n{n_sig}/{n_total} comparisons significant at alpha=0.05 (BH-corrected)")
print(f"\nSaved: {RESULTS / 'statistical_tests_summary.csv'}")
