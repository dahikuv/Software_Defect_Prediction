"""Artifact provenance helpers shared by scripts and app services."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.utils.coercion import coerce_bool

COMMIT_FEATURE_COUNT_KEYS = (
    "num_commit_features",
    "commit_feature_count",
    "tfidf_num_features",
    "sbert_num_features",
    "embedding_dim",
)

def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False

def _coerce_count(value: Any) -> int | None:
    if _is_missing(value):
        return None
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null", "false"}:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

def has_positive_commit_feature_count(payload: Mapping[str, Any]) -> bool:
    """Return True only when artifact metadata reports real commit-text features."""
    for key in COMMIT_FEATURE_COUNT_KEYS:
        count = _coerce_count(payload.get(key))
        if count is not None and count > 0:
            return True
    return False

def artifact_uses_commit_text(payload: Mapping[str, Any]) -> bool:
    """Resolve whether a model artifact actually uses commit text."""
    explicit = payload.get("uses_commit_text")
    return coerce_bool(explicit, default=False) or has_positive_commit_feature_count(payload)

__all__ = [
    "COMMIT_FEATURE_COUNT_KEYS",
    "artifact_uses_commit_text",
    "has_positive_commit_feature_count",
]
