"""Generate additional SHAP plots: beeswarm, bar, waterfall, force, dependence for each dataset."""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models.bundle import ModelBundle
from src.utils.io import read_csv
from src.utils.logging import get_logger
from src.utils.paths import PROCESSED_DATA_DIR, RESULTS_FIGURES_DIR, RESULTS_TABLES_DIR, SPLITS_DIR, ensure_project_dirs

matplotlib.use("Agg")
logger = get_logger(__name__)

FINAL_MODELS_PATH = RESULTS_TABLES_DIR / "final_models_by_dataset.csv"
BEST_MODELS_PATH = RESULTS_TABLES_DIR / "best_models_by_dataset.csv"

DEFAULT_METRICS = ["loc", "v(g)", "ev(g)", "iv(g)", "branchCount", "coupling", "cohesion", "code_churn"]
JITLINE_DATASETS = {"openstack", "qt", "jitfine"}

def load_best_models():
    path = FINAL_MODELS_PATH if FINAL_MODELS_PATH.exists() else BEST_MODELS_PATH
    return read_csv(path)

def load_split_frames(row):
    ds = str(row["dataset_name"])
    # Try processed parquet first
    parquet_path = PROCESSED_DATA_DIR / f"{ds}_clean.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        split_mode = str(row.get("split_mode", "saved_split") or "saved_split")
        if ds in JITLINE_DATASETS or split_mode == "jitline_native_split":
            norm = df["jitline_split"].astype(str).str.strip().str.lower()
            return df.loc[norm == "train"].copy(), df.loc[norm == "test"].copy()
    # Fallback to split row files
    split_dir = SPLITS_DIR / ds
    train_rows = split_dir / "train_rows.csv"
    test_rows = split_dir / "test_rows.csv"
    if train_rows.exists() and test_rows.exists():
        return read_csv(train_rows), read_csv(test_rows)
    raise FileNotFoundError(f"No data found for {ds}")

def get_feature_columns(df):
    exclude = {"label", "module_id", "project_name", "dataset_name", "commit_text", "jitline_split",
               "classification", "fix", "is_buggy_commit", "author_date", "author_name", "author_email",
               "parent_hashes", "commit_hash", "fileschanged", "author_date_unix_timestamp"}
    return [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64', 'int32', 'bool']]

def sanitize_names(X):
    rename_map = {}
    used = {}
    for c in X.columns:
        base = c.replace("(", "_").replace(")", "_").replace(" ", "_").replace("/", "_").replace("-", "_").strip("_") or "feature"
        count = used.get(base, 0)
        used[base] = count + 1
        rename_map[c] = base if count == 0 else f"{base}_{count + 1}"
    return X.rename(columns=rename_map), rename_map

def apply_map(X, rename_map):
    mapped = X.rename(columns=rename_map)
    ordered = [rename_map[c] for c in X.columns if c in rename_map]
    return mapped[ordered].copy()

def generate_plots(ds_name, model, X_train, X_test, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    n_bg = min(100, len(X_train))
    n_ex = min(50, len(X_test))
    X_bg = X_train.sample(n=n_bg, random_state=42)
    X_ex = X_test.sample(n=n_ex, random_state=42)

    logger.info("[%s] Computing SHAP values (bg=%d, explain=%d)", ds_name, len(X_bg), len(X_ex))
    explainer = shap.TreeExplainer(model, data=X_bg, feature_perturbation="interventional")
    sv = explainer.shap_values(X_ex)
    values = sv.values if hasattr(sv, 'values') else sv
    if isinstance(values, list):
        values = values[-1]
    if values.ndim == 3:
        values = values[:, :, -1]

    # 1. Beeswarm
    plt.figure(figsize=(10, 6))
    shap.summary_plot(values, X_ex, show=False, plot_type="dot")
    plt.tight_layout()
    plt.savefig(output_dir / f"{ds_name}_shap_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("[%s] beeswarm saved", ds_name)

    # 2. Bar
    plt.figure(figsize=(10, 6))
    shap.summary_plot(values, X_ex, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(output_dir / f"{ds_name}_shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("[%s] bar saved", ds_name)

    # 3. Waterfall
    try:
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = float(ev[-1]) if len(ev) > 1 else float(ev[0])
        else:
            ev = float(ev)
        explanation = shap.Explanation(
            values=values[0], base_values=ev,
            data=X_ex.iloc[0].values, feature_names=list(X_ex.columns),
        )
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / f"{ds_name}_shap_waterfall.png", dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("[%s] waterfall saved", ds_name)
    except Exception as e:
        logger.warning("[%s] waterfall failed: %s", ds_name, e)

    # 4. Force
    try:
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = float(ev[-1]) if len(ev) > 1 else float(ev[0])
        else:
            ev = float(ev)
        plt.figure()
        shap.force_plot(ev, values[0], X_ex.iloc[0], matplotlib=True, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / f"{ds_name}_shap_force.png", dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("[%s] force saved", ds_name)
    except Exception as e:
        logger.warning("[%s] force failed: %s", ds_name, e)

    # 5. Dependence (top feature)
    try:
        top_idx = int(np.argmax(np.abs(values).mean(axis=0)))
        top_feature = X_ex.columns[top_idx]
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(top_feature, values, X_ex, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / f"{ds_name}_shap_dependence.png", dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("[%s] dependence saved (%s)", ds_name, top_feature)
    except Exception as e:
        logger.warning("[%s] dependence failed: %s", ds_name, e)

def main():
    ensure_project_dirs()
    best_df = load_best_models()
    logger.info("Loaded %d best-model rows", len(best_df))

    for _, row in best_df.iterrows():
        ds = str(row["dataset_name"])
        model_path = Path(str(row["model_path"]))
        if not model_path.exists():
            logger.warning("[%s] Model not found: %s", ds, model_path)
            continue

        logger.info("[%s] Loading model from %s", ds, model_path.name)
        train_df, test_df = load_split_frames(row)

        loaded = joblib.load(model_path)
        if isinstance(loaded, ModelBundle):
            X_train = loaded.transform_features(train_df)
            X_test = loaded.transform_features(test_df)
            model = loaded.estimator
        else:
            feat_cols = get_feature_columns(train_df)
            X_train = train_df[feat_cols].copy()
            X_test = test_df[feat_cols].copy()
            model = loaded

        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        X_train, rename_map = sanitize_names(X_train)
        X_test = apply_map(X_test, rename_map)

        orig = getattr(model, "feature_names_in_", None)
        patched = False
        if orig is not None and len(orig) == len(X_train.columns):
            try:
                model.feature_names_in_ = np.asarray(list(X_train.columns), dtype=object)
                patched = True
            except (AttributeError, Exception):
                pass

        output_dir = RESULTS_FIGURES_DIR / "shap" / ds
        generate_plots(ds, model, X_train, X_test, output_dir)

        if patched:
            model.feature_names_in_ = orig

    logger.info("All SHAP extra plots generated.")

if __name__ == "__main__":
    main()
