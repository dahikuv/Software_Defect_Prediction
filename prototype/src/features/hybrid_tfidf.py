"""Train-fitted metrics + TF-IDF feature preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.commit_tfidf import normalize_commit_text
from src.features.metrics_features import MetricsFeatureSpec, fit_metrics_feature_spec, transform_metrics_features


@dataclass
class HybridTfidfFeatureSpec:
    """Train-fitted metrics + TF-IDF feature schema."""

    metrics_spec: MetricsFeatureSpec
    vectorizer: TfidfVectorizer
    text_column: str
    tfidf_feature_names: list[str]
    has_commit_text: bool

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        metrics_df = transform_metrics_features(df, self.metrics_spec)
        if not self.tfidf_feature_names:
            return metrics_df

        if self.text_column in df.columns:
            text_series = normalize_commit_text(df[self.text_column])
        else:
            text_series = pd.Series([""] * len(df), index=df.index)

        matrix = self.vectorizer.transform(text_series)
        tfidf_df = pd.DataFrame(
            matrix.toarray().astype("float32", copy=False),
            columns=[f"commit_{name}" for name in self.tfidf_feature_names],
            index=df.index,
        )
        # Both blocks are built with index=df.index, so concat aligns on that
        # shared index. Do NOT reset_index here: a non-contiguous source index
        # would otherwise misalign the metric and TF-IDF rows.
        return pd.concat([metrics_df, tfidf_df], axis=1)

    def to_metadata(self) -> dict:
        metadata = self.metrics_spec.to_metadata()
        uses_commit_text = bool(self.tfidf_feature_names)
        feature_family = "metrics_plus_commit_text" if uses_commit_text else "metrics_only"
        metadata.update(
            {
                "feature_family": feature_family,
                "feature_set": feature_family,
                "text_feature_column": self.text_column,
                "metrics_num_features": len(self.metrics_spec.selected_metrics),
                "tfidf_num_features": len(self.tfidf_feature_names),
                "tfidf_vocabulary_size": len(self.tfidf_feature_names),
                "has_commit_text": self.has_commit_text,
                "uses_commit_text": uses_commit_text,
            }
        )
        return metadata


def fit_hybrid_tfidf_feature_spec(
    df: pd.DataFrame,
    metrics: list[str],
    text_column: str = "commit_text",
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> HybridTfidfFeatureSpec:
    """Fit metrics imputation and TF-IDF vectorizer on a training frame."""
    metrics_spec = fit_metrics_feature_spec(df, metrics)
    if text_column in df.columns:
        text_series = normalize_commit_text(df[text_column])
    else:
        text_series = pd.Series([""] * len(df), index=df.index)

    has_commit_text = bool(text_series.str.split().str.len().fillna(0).gt(0).any())
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    tfidf_feature_names: list[str] = []
    if has_commit_text:
        try:
            vectorizer.fit(text_series)
            tfidf_feature_names = list(vectorizer.get_feature_names_out())
        except ValueError:
            tfidf_feature_names = []

    return HybridTfidfFeatureSpec(
        metrics_spec=metrics_spec,
        vectorizer=vectorizer,
        text_column=text_column,
        tfidf_feature_names=tfidf_feature_names,
        has_commit_text=has_commit_text,
    )


def build_hybrid_tfidf_training_frame(
    df: pd.DataFrame,
    spec: HybridTfidfFeatureSpec,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Transform a labeled frame using a train-fitted hybrid TF-IDF spec."""
    if "label" not in df.columns:
        raise ValueError("The input DataFrame must contain a 'label' column.")
    X = spec.transform(df)
    y = pd.to_numeric(df["label"], errors="coerce")
    if y.isna().any():
        raise ValueError("The 'label' column contains non-numeric values after preprocessing.")
    metadata = spec.to_metadata()
    metadata["num_rows"] = len(df)
    metadata["num_features"] = int(X.shape[1])
    metadata["label_distribution"] = y.value_counts().to_dict()
    return X, y.astype(int), metadata
