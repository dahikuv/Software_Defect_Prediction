"""Analyze the impact of commit-message features versus metrics-only baselines."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from pandas.errors import EmptyDataError

from src.evaluation.compare import build_results_table
from src.utils.coercion import coerce_bool
from src.utils.io import read_csv, write_csv, write_json
from src.utils.logging import get_logger
from src.utils.paths import RESULTS_TABLES_DIR, ensure_project_dirs

logger = get_logger(__name__)

BASELINE_RESULTS_PATH = RESULTS_TABLES_DIR / "results_table.csv"
TEXT_BRANCH_SOURCES = {
    "tfidf": {
        "results": RESULTS_TABLES_DIR / "hybrid_tfidf_results.csv",
        "coverage": RESULTS_TABLES_DIR / "hybrid_tfidf_feature_coverage.csv",
    },
    "sbert": {
        "results": RESULTS_TABLES_DIR / "hybrid_sbert_results.csv",
        "coverage": RESULTS_TABLES_DIR / "hybrid_sbert_feature_coverage.csv",
    },
}
ERROR_CASES_PATH = RESULTS_TABLES_DIR / "error_analysis_cases.csv"

IMPACT_TABLE_PATH = RESULTS_TABLES_DIR / "commit_message_impact.csv"
IMPACT_SUMMARY_PATH = RESULTS_TABLES_DIR / "commit_message_impact_summary.csv"
IMPACT_DELTA_PATH = RESULTS_TABLES_DIR / "commit_message_impact_deltas.csv"
IMPACT_META_PATH = RESULTS_TABLES_DIR / "commit_message_impact_meta.json"
IMPACT_HITS_PATH = RESULTS_TABLES_DIR / "commit_message_impact_improvements.csv"
IMPACT_FIGURE_PATH = RESULTS_TABLES_DIR / "commit_message_impact_deltas.png"
IMPACT_BRANCH_FIGURE_PATH = RESULTS_TABLES_DIR / "commit_message_impact_branch_means.png"

METRICS = ["accuracy", "precision", "recall", "f1", "auc"]
IMPACT_COLUMNS = [
    "dataset_name",
    "model",
    "text_branch",
    "baseline_feature_family",
    "hybrid_feature_family",
    "commit_text_source",
    "uses_commit_text_baseline",
    "uses_commit_text_hybrid",
    *[f"delta_{metric}" for metric in METRICS],
    *[f"relative_delta_{metric}" for metric in METRICS],
    "improved_f1",
    "improved_auc",
]
IMPACT_SUMMARY_COLUMNS = [
    "text_branch",
    "dataset_name",
    "num_models",
    "mean_delta_accuracy",
    "mean_delta_precision",
    "mean_delta_recall",
    "mean_delta_f1",
    "mean_delta_auc",
    "num_improved_f1",
    "num_improved_auc",
]
BRANCH_SUMMARY_COLUMNS = [
    "text_branch",
    "num_datasets",
    "num_models",
    "mean_delta_accuracy",
    "mean_delta_precision",
    "mean_delta_recall",
    "mean_delta_f1",
    "mean_delta_auc",
    "num_improved_f1",
    "num_improved_auc",
]


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    return build_results_table(df.to_dict("records")) if not df.empty else pd.DataFrame()


def _normalize_branch_name(feature_mode: str | None) -> str:
    if not feature_mode:
        return "unknown"
    lowered = feature_mode.lower()
    if "tfidf" in lowered:
        return "tfidf"
    if "sbert" in lowered:
        return "sbert"
    return lowered


def _available_text_branches() -> dict[str, pd.DataFrame]:
    branches: dict[str, pd.DataFrame] = {}
    for branch_name, paths in TEXT_BRANCH_SOURCES.items():
        results_df = _load_table(paths["results"])
        if not results_df.empty:
            filtered_df = results_df.copy()
            if "uses_commit_text" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["uses_commit_text"].map(coerce_bool)]
            feature_count_columns = [column for column in ["tfidf_num_features", "sbert_num_features", "embedding_dim"] if column in filtered_df.columns]
            if feature_count_columns:
                has_text_features = pd.Series(False, index=filtered_df.index)
                for column in feature_count_columns:
                    has_text_features = has_text_features | (pd.to_numeric(filtered_df[column], errors="coerce").fillna(0) > 0)
                filtered_df = filtered_df[has_text_features]
            if not filtered_df.empty:
                branches[branch_name] = filtered_df
    return branches


def _branch_alignment_key(df: pd.DataFrame) -> list[str]:
    return [col for col in ["dataset_name", "model"] if col in df.columns]


def _build_comparison_for_branch(baseline_df: pd.DataFrame, branch_name: str, branch_df: pd.DataFrame) -> pd.DataFrame:
    if baseline_df.empty or branch_df.empty:
        return pd.DataFrame()

    key_cols = _branch_alignment_key(baseline_df)
    if not key_cols:
        return pd.DataFrame()

    metadata_cols = [
        "feature_family",
        "feature_set",
        "text_feature_column",
        "uses_commit_text",
        "tfidf_num_features",
        "sbert_num_features",
        "embedding_dim",
        "num_commit_features",
    ]
    baseline_cols = key_cols + [c for c in METRICS + metadata_cols if c in baseline_df.columns]
    branch_cols = key_cols + [c for c in METRICS + metadata_cols if c in branch_df.columns]
    baseline = baseline_df[baseline_cols].copy()
    branch = branch_df[branch_cols].copy()

    merged = baseline.merge(branch, on=key_cols, suffixes=("_baseline", f"_{branch_name}"), how="inner")
    if merged.empty:
        return merged

    merged["text_branch"] = branch_name
    merged["baseline_feature_family"] = merged.get("feature_family_baseline", "metrics_only")
    merged["hybrid_feature_family"] = merged.get("feature_family_" + branch_name, branch_name)
    if f"text_feature_column_{branch_name}" in merged.columns:
        merged["commit_text_source"] = merged[f"text_feature_column_{branch_name}"]
    if "uses_commit_text_baseline" in merged.columns:
        merged["uses_commit_text_baseline"] = merged["uses_commit_text_baseline"].map(coerce_bool)
    else:
        merged["uses_commit_text_baseline"] = False
    if f"uses_commit_text_{branch_name}" in merged.columns:
        merged["uses_commit_text_hybrid"] = merged[f"uses_commit_text_{branch_name}"].map(coerce_bool)
    else:
        merged["uses_commit_text_hybrid"] = False

    for metric in METRICS:
        b = f"{metric}_baseline"
        h = f"{metric}_{branch_name}"
        if b in merged.columns and h in merged.columns:
            merged[f"delta_{metric}"] = merged[h] - merged[b]
            merged[f"relative_delta_{metric}"] = merged.apply(
                lambda row: ((row[h] - row[b]) / row[b]) if pd.notna(row[b]) and row[b] not in (0, 0.0) else pd.NA,
                axis=1,
            )

    merged["improved_f1"] = merged.get("delta_f1", pd.Series(dtype=float)).fillna(0) > 0
    merged["improved_auc"] = merged.get("delta_auc", pd.Series(dtype=float)).fillna(0) > 0
    return merged


def _build_all_comparisons(baseline_df: pd.DataFrame, branches: dict[str, pd.DataFrame]) -> pd.DataFrame:
    comparisons: list[pd.DataFrame] = []
    for branch_name, branch_df in branches.items():
        comparison_df = _build_comparison_for_branch(baseline_df, branch_name, branch_df)
        if not comparison_df.empty:
            comparisons.append(comparison_df)
    if not comparisons:
        return pd.DataFrame()
    return pd.concat(comparisons, ignore_index=True, sort=False)


def _summarize_impact(impact_df: pd.DataFrame) -> pd.DataFrame:
    if impact_df.empty:
        return impact_df
    summary = (
        impact_df.groupby(["text_branch", "dataset_name"], as_index=False)
        .agg(
            num_models=("model", "count"),
            mean_delta_accuracy=("delta_accuracy", "mean"),
            mean_delta_precision=("delta_precision", "mean"),
            mean_delta_recall=("delta_recall", "mean"),
            mean_delta_f1=("delta_f1", "mean"),
            mean_delta_auc=("delta_auc", "mean"),
            num_improved_f1=("improved_f1", "sum"),
            num_improved_auc=("improved_auc", "sum"),
        )
        .sort_values(["text_branch", "mean_delta_auc", "mean_delta_f1"], ascending=[True, False, False])
    )
    return summary


def _build_branch_summary(impact_df: pd.DataFrame) -> pd.DataFrame:
    if impact_df.empty:
        return impact_df
    return (
        impact_df.groupby("text_branch", as_index=False)
        .agg(
            num_datasets=("dataset_name", "nunique"),
            num_models=("model", "count"),
            mean_delta_accuracy=("delta_accuracy", "mean"),
            mean_delta_precision=("delta_precision", "mean"),
            mean_delta_recall=("delta_recall", "mean"),
            mean_delta_f1=("delta_f1", "mean"),
            mean_delta_auc=("delta_auc", "mean"),
            num_improved_f1=("improved_f1", "sum"),
            num_improved_auc=("improved_auc", "sum"),
        )
        .sort_values(["mean_delta_auc", "mean_delta_f1"], ascending=[False, False])
    )


def _build_improvement_table(impact_df: pd.DataFrame) -> pd.DataFrame:
    if impact_df.empty:
        return impact_df
    improvements = impact_df.loc[(impact_df["improved_f1"] == True) | (impact_df["improved_auc"] == True)].copy()  # noqa: E712
    if improvements.empty:
        return improvements
    return improvements.sort_values(["text_branch", "delta_auc", "delta_f1"], ascending=[True, False, False])


def _plot_impact_deltas(impact_df: pd.DataFrame, output_path: Path) -> None:
    if impact_df.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional in minimal envs
        logger.warning("Matplotlib unavailable; skipping impact figure: %s", exc)
        return

    branch_order = list(dict.fromkeys(impact_df["text_branch"].tolist()))
    fig, axes = plt.subplots(len(branch_order), 1, figsize=(10, max(3, 3 * len(branch_order))), constrained_layout=True)
    if len(branch_order) == 1:
        axes = [axes]

    for ax, branch_name in zip(axes, branch_order):
        branch_df = impact_df[impact_df["text_branch"] == branch_name].copy()
        branch_df = branch_df.sort_values(["dataset_name", "model"])
        labels = branch_df.apply(lambda row: f"{row['dataset_name']}:{row['model']}", axis=1).tolist()
        ax.axhline(0, color="black", linewidth=0.8)
        ax.plot(labels, branch_df["delta_f1"].tolist(), marker="o", label="ΔF1")
        ax.plot(labels, branch_df["delta_auc"].tolist(), marker="s", label="ΔAUC")
        ax.set_title(f"Commit-message impact vs baseline — {branch_name}")
        ax.set_ylabel("Delta")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_branch_means(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional in minimal envs
        logger.warning("Matplotlib unavailable; skipping branch-means figure: %s", exc)
        return

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    x = range(len(summary_df))
    ax.bar([i - 0.15 for i in x], summary_df["mean_delta_f1"].tolist(), width=0.3, label="Mean ΔF1")
    ax.bar([i + 0.15 for i in x], summary_df["mean_delta_auc"].tolist(), width=0.3, label="Mean ΔAUC")
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary_df["text_branch"].tolist())
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Delta")
    ax.set_title("Mean impact by text branch")
    ax.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_project_dirs()

    baseline_df = _load_table(BASELINE_RESULTS_PATH)
    coverage_frames = []
    for branch_name, paths in TEXT_BRANCH_SOURCES.items():
        coverage_path = paths["coverage"]
        if coverage_path.exists():
            coverage_df = read_csv(coverage_path)
            if not coverage_df.empty:
                coverage_df = coverage_df.copy()
                coverage_df["text_branch"] = branch_name
                coverage_frames.append(coverage_df)
    error_cases_df = read_csv(ERROR_CASES_PATH) if ERROR_CASES_PATH.exists() else pd.DataFrame()

    if baseline_df.empty:
        raise FileNotFoundError(f"Missing baseline results table: {BASELINE_RESULTS_PATH}")

    branches = _available_text_branches()
    impact_df = _build_all_comparisons(baseline_df, branches)
    impact_status = "available" if not impact_df.empty else "unsupported_no_valid_commit_text_branch"
    impact_message = (
        "Commit-message impact was computed from valid text-feature branches."
        if impact_status == "available"
        else "Commit-message impact unavailable because no valid aligned commit-text feature branch was available."
    )
    summary_df = _summarize_impact(impact_df)
    branch_summary_df = _build_branch_summary(impact_df)
    improvements_df = _build_improvement_table(impact_df)

    delta_columns = ["dataset_name", "model", "text_branch"] + [c for c in impact_df.columns if c.startswith("delta_") or c.startswith("relative_delta_")]
    delta_columns = [c for c in delta_columns if c in impact_df.columns]
    delta_df = impact_df[delta_columns].copy() if not impact_df.empty else impact_df
    if impact_df.empty and len(impact_df.columns) == 0:
        impact_df = pd.DataFrame(columns=IMPACT_COLUMNS)
    if summary_df.empty and len(summary_df.columns) == 0:
        summary_df = pd.DataFrame(columns=IMPACT_SUMMARY_COLUMNS)
    if branch_summary_df.empty and len(branch_summary_df.columns) == 0:
        branch_summary_df = pd.DataFrame(columns=BRANCH_SUMMARY_COLUMNS)
    if improvements_df.empty and len(improvements_df.columns) == 0:
        improvements_df = pd.DataFrame(columns=IMPACT_COLUMNS)
    if delta_df.empty and len(delta_df.columns) == 0:
        delta_df = pd.DataFrame(columns=["dataset_name", "model", "text_branch", *[f"delta_{metric}" for metric in METRICS], *[f"relative_delta_{metric}" for metric in METRICS]])

    write_csv(impact_df, IMPACT_TABLE_PATH)
    write_csv(summary_df, IMPACT_SUMMARY_PATH)
    write_csv(delta_df, IMPACT_DELTA_PATH)
    write_csv(improvements_df, IMPACT_HITS_PATH)
    write_csv(branch_summary_df, RESULTS_TABLES_DIR / "commit_message_impact_branch_summary.csv")

    _plot_impact_deltas(impact_df, IMPACT_FIGURE_PATH)
    _plot_branch_means(branch_summary_df, IMPACT_BRANCH_FIGURE_PATH)

    write_json(
        {
            "baseline_results_table": str(BASELINE_RESULTS_PATH),
            "impact_status": impact_status,
            "impact_message": impact_message,
            "applicable_dataset_family": "commit_level",
            "applicable_datasets": ["openstack", "qt", "jitfine"],
            "available_text_branches": {
                branch_name: {
                    "results": str(paths["results"]),
                    "coverage": str(paths["coverage"]),
                }
                for branch_name, paths in TEXT_BRANCH_SOURCES.items()
                if branch_name in branches
            },
            "impact_table": str(IMPACT_TABLE_PATH),
            "impact_summary": str(IMPACT_SUMMARY_PATH),
            "impact_deltas": str(IMPACT_DELTA_PATH),
            "impact_improvements": str(IMPACT_HITS_PATH),
            "branch_summary": str(RESULTS_TABLES_DIR / "commit_message_impact_branch_summary.csv"),
            "impact_figure": str(IMPACT_FIGURE_PATH),
            "branch_figure": str(IMPACT_BRANCH_FIGURE_PATH),
            "error_cases_table": str(ERROR_CASES_PATH),
            "coverage_sources": [
                {"text_branch": branch_name, "path": str(paths["coverage"])}
                for branch_name, paths in TEXT_BRANCH_SOURCES.items()
                if branch_name in branches
            ],
            "artifact_schema_version": "paper-v1",
            "applicable_dataset_family": "commit_level",
            "applicable_datasets": ["openstack", "qt", "jitfine"],
            "scope_note": "Commit-message impact is computed only for commit-level datasets that contain commit_text. PROMISE module-level datasets are excluded by design.",
        },
        IMPACT_META_PATH,
    )

    logger.info("Saved impact table to %s", IMPACT_TABLE_PATH)
    logger.info("Saved impact summary to %s", IMPACT_SUMMARY_PATH)
    logger.info("Saved impact deltas to %s", IMPACT_DELTA_PATH)
    logger.info("Saved impact improvements to %s", IMPACT_HITS_PATH)
    logger.info("Saved branch summary to %s", RESULTS_TABLES_DIR / "commit_message_impact_branch_summary.csv")
    logger.info("Saved impact figure to %s", IMPACT_FIGURE_PATH)
    logger.info("Saved branch figure to %s", IMPACT_BRANCH_FIGURE_PATH)
    logger.info("Saved impact metadata to %s", IMPACT_META_PATH)
    if impact_status != "available":
        logger.warning(impact_message)


if __name__ == "__main__":
    main()
