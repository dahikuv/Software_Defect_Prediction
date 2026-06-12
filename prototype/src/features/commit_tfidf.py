"""TF-IDF feature pipeline for commit messages."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_commit_text(series: pd.Series) -> pd.Series:
    """Apply lightweight normalization to commit text."""
    return series.fillna("").astype(str).str.lower().str.strip()


def build_tfidf_features(
    text_series: pd.Series,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> Tuple[TfidfVectorizer, pd.DataFrame]:
    """Fit a TF-IDF vectorizer and return dense feature DataFrame."""
    normalized = normalize_commit_text(text_series)
    usable_mask = normalized.str.split().str.len().fillna(0).gt(0)
    if not usable_mask.any():
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        return vectorizer, pd.DataFrame(index=text_series.index)

    usable_text = normalized.loc[usable_mask]
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    try:
        matrix = vectorizer.fit_transform(usable_text)
    except ValueError:
        return vectorizer, pd.DataFrame(index=text_series.index)

    usable_features = pd.DataFrame(
        matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=usable_text.index,
    )
    features = pd.DataFrame(index=text_series.index).join(usable_features, how="left").fillna(0.0)
    return vectorizer, features
