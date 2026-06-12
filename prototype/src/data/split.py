"""Train/validation/test split utilities."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import read_csv, write_csv, write_json


def _can_stratify(series: pd.Series) -> bool:
    counts = series.value_counts(dropna=False)
    return len(counts) >= 2 and counts.min() >= 2


def stratified_split(
    df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a labeled DataFrame into train/val/test partitions."""
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1")

    stratify_series = df[label_col] if _can_stratify(df[label_col]) else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify_series,
        random_state=random_state,
    )

    relative_val_size = val_size / (1 - test_size)
    if not 0 < relative_val_size < 1:
        raise ValueError("Computed validation split ratio is invalid")

    train_stratify = train_df[label_col] if _can_stratify(train_df[label_col]) else None
    train_df, val_df = train_test_split(
        train_df,
        test_size=relative_val_size,
        stratify=train_stratify,
        random_state=random_state,
    )
    return train_df, val_df, test_df


def build_split_record(dataset_name: str, split_name: str, df: pd.DataFrame, label_col: str = "label") -> dict[str, Any]:
    """Build a compact split summary record."""
    counts = df[label_col].value_counts().to_dict() if label_col in df.columns else {}
    return {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "num_rows": int(len(df)),
        "num_clean": int(counts.get(0, 0)),
        "num_defective": int(counts.get(1, 0)),
    }


def validate_split_ids(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, id_col: str = "module_id") -> None:
    """Validate split IDs for uniqueness and coverage."""
    for frame, split_name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        if id_col not in frame.columns:
            raise ValueError(f"Missing identifier column in {split_name} split: {id_col}")
        if frame[id_col].isna().any():
            raise ValueError(f"Missing values found in {split_name} split identifier column: {id_col}")

    combined = pd.concat(
        [
            train_df[[id_col]].assign(split_name="train"),
            val_df[[id_col]].assign(split_name="val"),
            test_df[[id_col]].assign(split_name="test"),
        ],
        ignore_index=True,
    )
    if combined[id_col].duplicated().any():
        duplicates = combined.loc[combined[id_col].duplicated(), id_col].tolist()[:10]
        raise ValueError(f"Duplicate split identifiers found: {duplicates}")


def split_manifest_hash(manifest: dict[str, Any]) -> str:
    """Return a stable hash for a split manifest."""
    manifest_copy = dict(manifest)
    manifest_copy.pop("manifest_hash", None)
    payload = json.dumps(manifest_copy, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha256(payload).hexdigest()


def build_split_manifest(
    dataset_name: str,
    source_file: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_seed: int,
    test_size: float,
    val_size: float,
    stratify_enabled: bool,
    id_col: str = "module_id",
) -> dict[str, Any]:
    """Build a reproducible split manifest for one dataset."""
    manifest: dict[str, Any] = {
        "dataset_name": dataset_name,
        "source_file": source_file,
        "id_col": id_col,
        "random_seed": int(random_seed),
        "test_size": float(test_size),
        "val_size": float(val_size),
        "stratify_enabled": bool(stratify_enabled),
        "num_train_rows": int(len(train_df)),
        "num_val_rows": int(len(val_df)),
        "num_test_rows": int(len(test_df)),
        "train_num_clean": int(train_df["label"].value_counts().to_dict().get(0, 0)) if "label" in train_df.columns else 0,
        "train_num_defective": int(train_df["label"].value_counts().to_dict().get(1, 0)) if "label" in train_df.columns else 0,
        "val_num_clean": int(val_df["label"].value_counts().to_dict().get(0, 0)) if "label" in val_df.columns else 0,
        "val_num_defective": int(val_df["label"].value_counts().to_dict().get(1, 0)) if "label" in val_df.columns else 0,
        "test_num_clean": int(test_df["label"].value_counts().to_dict().get(0, 0)) if "label" in test_df.columns else 0,
        "test_num_defective": int(test_df["label"].value_counts().to_dict().get(1, 0)) if "label" in test_df.columns else 0,
        "train_ids_path": f"splits/{dataset_name}/train_ids.csv",
        "val_ids_path": f"splits/{dataset_name}/val_ids.csv",
        "test_ids_path": f"splits/{dataset_name}/test_ids.csv",
    }
    manifest["manifest_hash"] = split_manifest_hash(manifest)
    return manifest


def save_split_ids(df: pd.DataFrame, path: str | Path, id_col: str = "module_id") -> None:
    """Save split row identifiers for reproducibility."""
    if id_col not in df.columns:
        raise ValueError(f"Missing identifier column: {id_col}")
    write_csv(df[[id_col]].copy(), path)


def save_split_summary(records: list[dict[str, Any]], path: str | Path) -> None:
    """Save split summary records to CSV."""
    write_csv(pd.DataFrame(records), path)


def save_split_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    """Save a split manifest as JSON."""
    write_json(manifest, path)


def load_split_ids(path: str | Path, id_col: str = "module_id") -> pd.Series:
    """Load split identifiers from CSV."""
    df = read_csv(path)
    if id_col not in df.columns:
        raise ValueError(f"Missing identifier column in split file: {id_col}")
    return df[id_col]


def reconstruct_split_frames(
    cleaned_df: pd.DataFrame,
    train_ids_path: str | Path,
    val_ids_path: str | Path,
    test_ids_path: str | Path,
    id_col: str = "module_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild train/val/test frames from saved split IDs."""
    if id_col not in cleaned_df.columns:
        raise ValueError(f"Missing identifier column in cleaned dataset: {id_col}")

    train_ids = set(load_split_ids(train_ids_path, id_col=id_col).astype(str).tolist())
    val_ids = set(load_split_ids(val_ids_path, id_col=id_col).astype(str).tolist())
    test_ids = set(load_split_ids(test_ids_path, id_col=id_col).astype(str).tolist())

    all_ids = train_ids | val_ids | test_ids
    if len(all_ids) != len(train_ids) + len(val_ids) + len(test_ids):
        raise ValueError("Duplicate identifiers detected across split artifacts")

    available_ids = set(cleaned_df[id_col].astype(str).tolist())
    missing_ids = all_ids - available_ids
    if missing_ids:
        raise ValueError(f"Split artifacts reference missing dataset IDs: {sorted(list(missing_ids))[:10]}")

    train_df = cleaned_df[cleaned_df[id_col].astype(str).isin(train_ids)].copy()
    val_df = cleaned_df[cleaned_df[id_col].astype(str).isin(val_ids)].copy()
    test_df = cleaned_df[cleaned_df[id_col].astype(str).isin(test_ids)].copy()

    if len(train_df) + len(val_df) + len(test_df) != len(all_ids):
        raise ValueError("Split reconstruction did not cover all rows exactly once")

    return train_df, val_df, test_df


def build_split_index(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a top-level index for split artifacts."""
    return pd.DataFrame(records)


def split_with_ids(df: pd.DataFrame, label_col: str = "label", test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compatibility helper that returns train/val/test splits."""
    return stratified_split(df, label_col=label_col, test_size=test_size, val_size=val_size, random_state=random_state)
