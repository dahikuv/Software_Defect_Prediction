"""
Generate hyperparameter tuning comparison table.
Shows baseline (default) vs tuned performance for the selected model per dataset.

Output: results/tables/hyperparam_comparison.csv
"""

import pandas as pd
import sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

RESULTS = Path(__file__).resolve().parent.parent / "results" / "tables"

# Selected model per dataset (from Table 2 in the paper)
SELECTED = {
    "cm1": "xgb", "jm1": "xgb", "kc1": "xgb", "pc1": "xgb",
    "jitfine": "rf", "openstack": "lgbm", "qt": "xgb",
}

# Baseline: default hyperparameters from results_table.csv
baseline_df = pd.read_csv(RESULTS / "results_table.csv")
# Tuned: from baseline_vs_tuned_comparison.csv
tuned_df = pd.read_csv(RESULTS / "baseline_vs_tuned_comparison.csv")
tuned_rows = tuned_df[tuned_df["artifact_stage"] == "tuned_best_table"].copy()

rows = []
for ds, model in SELECTED.items():
    # Baseline (default) results
    b_row = baseline_df[
        (baseline_df["dataset_name"] == ds) & (baseline_df["model"] == model)
    ]
    if b_row.empty:
        continue
    b_row = b_row.iloc[0]

    # Tuned results
    t_row = tuned_rows[
        (tuned_rows["dataset_name"] == ds) & (tuned_rows["model"] == model)
    ]
    if t_row.empty:
        continue
    t_row = t_row.iloc[0]

    b_acc = b_row["accuracy"]
    b_prec = b_row["precision"]
    b_rec = b_row["recall"]
    b_f1 = b_row["f1"]
    b_auc = b_row["auc"]

    t_acc = t_row["tuned_accuracy"]
    t_prec = t_row["tuned_precision"]
    t_rec = t_row["tuned_recall"]
    t_f1 = t_row["tuned_f1"]
    t_auc = t_row["tuned_auc"]

    rows.append({
        "dataset": ds.upper(),
        "model": model.upper(),
        "baseline_acc": b_acc, "tuned_acc": t_acc, "delta_acc": t_acc - b_acc,
        "baseline_prec": b_prec, "tuned_prec": t_prec, "delta_prec": t_prec - b_prec,
        "baseline_rec": b_rec, "tuned_rec": t_rec, "delta_rec": t_rec - b_rec,
        "baseline_f1": b_f1, "tuned_f1": t_f1, "delta_f1": t_f1 - b_f1,
        "baseline_auc": b_auc, "tuned_auc": t_auc, "delta_auc": t_auc - b_auc,
    })

out = pd.DataFrame(rows)
out.to_csv(RESULTS / "hyperparam_comparison.csv", index=False)

# Pretty print
print("=== Hyperparameter Tuning Comparison ===")
for _, r in out.iterrows():
    print(f"\n{r['dataset']} ({r['model']}):")
    print(f"  {'Metric':<8} {'Baseline':>9} {'Tuned':>9} {'Delta':>9}")
    for m_name, m_key in [("Acc", "acc"), ("Prec", "prec"), ("Rec", "rec"), ("F1", "f1"), ("AUC", "auc")]:
        b = r[f"baseline_{m_key}"]
        t = r[f"tuned_{m_key}"]
        d = r[f"delta_{m_key}"]
        print(f"  {m_name:<8} {b:>9.3f} {t:>9.3f} {d:>+9.3f}")

print(f"\nSaved: {RESULTS / 'hyperparam_comparison.csv'}")
