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

        if "module_id" in merged.columns and "module_id" in block.columns:
            block_for_merge = block.drop(
                columns=[key for key in MERGE_KEY_CANDIDATES if key != "module_id" and key in block.columns],
                errors="ignore",
            )
            if block_for_merge["module_id"].duplicated().any():
                raise ValueError(
                    "Feature block has duplicate 'module_id' values; a left merge would "
                    "fan out and duplicate base rows. Deduplicate the feature block first."
                )
            n_before = len(merged)
            merged = merged.merge(block_for_merge, on="module_id", how="left")
            if len(merged) != n_before:
                raise ValueError(
                    f"Row count changed during module_id merge ({n_before} -> {len(merged)}); "
                    "merge keys are not unique."
                )
            continue

        if len(block) != len(merged):
            raise ValueError(
                f"Cannot align feature block by position: base has {len(merged)} rows "
                f"but block has {len(block)} rows, and no shared 'module_id' key is available."
            )
        block_indexed = block.copy()
        block_indexed.index = merged.index
        merged = pd.concat([merged, block_indexed], axis=1)

    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged
