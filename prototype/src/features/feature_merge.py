"""Helpers for merging metrics and commit-derived features."""

from __future__ import annotations

import pandas as pd


MERGE_KEY_CANDIDATES = ["module_id", "project_name"]


def _prepare_feature_block(feature_df: pd.DataFrame | None, prefix: str) -> pd.DataFrame:
    """Return a safe feature block with prefixed non-key columns."""
    if feature_df is None or feature_df.empty:
        return pd.DataFrame()

    block = feature_df.copy()
    rename_map = {
        column: f"{prefix}{column}"
        for column in block.columns
        if column not in MERGE_KEY_CANDIDATES
    }
    return block.rename(columns=rename_map)


def merge_feature_sets(
    base_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
    text_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge base rows with metric and commit-derived feature blocks.

    Priority:
    - merge by `module_id` when available in both frames
    - keep `project_name` available for future grouped joins
    - fall back to index alignment only when key-based merging is not possible
    """
    merged = base_df.copy()

    metric_block = _prepare_feature_block(metrics_df, prefix="metric_")
    text_block = _prepare_feature_block(text_df, prefix="commit_")

    for block in [metric_block, text_block]:
        if block.empty:
            continue

        can_merge_on_module = "module_id" in merged.columns and "module_id" in block.columns
        if can_merge_on_module:
            merged = merged.merge(block, on=[key for key in MERGE_KEY_CANDIDATES if key in merged.columns and key in block.columns], how="left")
            continue

        block_indexed = block.reset_index(drop=True)
        merged = pd.concat([merged.reset_index(drop=True), block_indexed], axis=1)

    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged
