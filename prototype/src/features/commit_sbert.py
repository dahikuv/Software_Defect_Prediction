"""Sentence-BERT feature builder for commit messages."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_sbert_features(
    text_series: pd.Series,
    model_name: str = "all-MiniLM-L6-v2",
    normalize_embeddings: bool = True,
    batch_size: int = 32,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Encode commit messages into SBERT embeddings.

    If sentence-transformers is not available or text_series is empty, returns
    an empty feature frame with the same index. This keeps the function safe for
    lightweight environments while still supporting the full embedding path when
    the dependency is installed.
    """
    cleaned = text_series.fillna("").astype(str).str.strip()
    metadata: dict[str, Any] = {
        "model_name": model_name,
        "num_rows": int(len(cleaned)),
        "embedding_dim": 0,
        "used_fallback": False,
        "normalize_embeddings": bool(normalize_embeddings),
        "batch_size": int(batch_size),
    }

    if cleaned.empty:
        empty = pd.DataFrame(index=text_series.index)
        return (empty, metadata) if return_metadata else empty

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        empty = pd.DataFrame(index=text_series.index)
        metadata["used_fallback"] = True
        metadata["notes"] = "sentence-transformers is not installed"
        return (empty, metadata) if return_metadata else empty

    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            cleaned.tolist(),
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )
    except Exception as exc:  # pragma: no cover
        empty = pd.DataFrame(index=text_series.index)
        metadata["used_fallback"] = True
        metadata["notes"] = str(exc)
        return (empty, metadata) if return_metadata else empty

    embedding_df = pd.DataFrame(embeddings, index=text_series.index)
    embedding_df.columns = [f"sbert_{idx}" for idx in range(embedding_df.shape[1])]
    metadata["embedding_dim"] = int(embedding_df.shape[1])
    metadata["feature_columns"] = list(embedding_df.columns)
    return (embedding_df, metadata) if return_metadata else embedding_df
