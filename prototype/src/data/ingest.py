"""Dataset ingestion interfaces for raw software defect datasets."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.clean import clean_dataset
from src.data.unify_schema import unify_schema
from src.data.validate import validate_dataset_schema
from src.utils.paths import CONFIG_PATH

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json", ".arff"}
TEXT_COLUMN_CANDIDATES = {"commit_text", "commit_message", "commit_msg", "message", "log", "commit"}
LABEL_COLUMN_CANDIDATES = {"label", "bug", "bugs", "defect", "defects", "is_buggy", "is_defective"}


def load_dataset_selection() -> dict[str, set[str]]:
    """Load dataset selection tiers from project config."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    selection = config.get("data", {}).get("selection", {})
    return {
        "primary_files": {name.strip().lower().replace("\\", "/") for name in selection.get("primary_files", [])},
        "primary_datasets": {name.strip().lower() for name in selection.get("primary_datasets", [])},
        "supplementary": {name.strip().lower() for name in selection.get("supplementary", [])},
        "excluded": {name.strip().lower() for name in selection.get("excluded", [])},
    }


def normalize_config_path_to_absolute(path: str | Path) -> Path:
    """Resolve a config-listed dataset path relative to the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (CONFIG_PATH.parents[2] / candidate).resolve()


def _normalize_selection_path(path: str | Path) -> str:
    return str(Path(path).resolve()).strip().lower().replace("\\", "/")


def selected_primary_source_files() -> set[str]:
    """Return normalized absolute paths for the configured primary files."""
    selection = load_dataset_selection()
    return {
        _normalize_selection_path(normalize_config_path_to_absolute(path))
        for path in selection["primary_files"]
    }


def classify_dataset(dataset_name: str, source_file: str | Path | None = None) -> str:
    """Classify a dataset as primary, supplementary, excluded, candidate_primary, or unknown."""
    normalized_name = dataset_name.strip().lower()
    selection = load_dataset_selection()

    if source_file is not None and _normalize_selection_path(source_file) in selected_primary_source_files():
        return "primary"
    if normalized_name in selection["excluded"]:
        return "excluded"
    if normalized_name in selection["supplementary"]:
        return "supplementary"
    if normalized_name in selection["primary_datasets"]:
        return "candidate_primary"
    return "unknown"


def classify_dataset_file(file_path: str | Path) -> tuple[str, bool, bool, bool, str]:
    """Return selection metadata for a discovered dataset file."""
    path_obj = Path(file_path).resolve()
    dataset_tier = classify_dataset(path_obj.stem, path_obj)
    is_primary = dataset_tier == "primary"
    is_supplementary = dataset_tier == "supplementary"
    selected_for_baseline = is_primary

    if dataset_tier == "primary":
        selection_reason = "selected_primary_file"
    elif dataset_tier == "candidate_primary":
        selection_reason = "same_dataset_name_not_selected_file"
    elif dataset_tier == "supplementary":
        selection_reason = "supplementary_dataset"
    elif dataset_tier == "excluded":
        selection_reason = "excluded_dataset"
    else:
        selection_reason = "unclassified_dataset"

    return dataset_tier, is_primary, is_supplementary, selected_for_baseline, selection_reason


def selection_note(file_path: str | Path) -> str:
    """Return a concise note describing dataset selection status."""
    dataset_tier, _, _, selected_for_baseline, selection_reason = classify_dataset_file(file_path)
    return f"tier={dataset_tier}; selected_for_baseline={selected_for_baseline}; selection={selection_reason}"


def selected_for_baseline(dataset_name: str, source_file: str | Path) -> bool:
    """Return True if the dataset row belongs to the final baseline set."""
    return classify_dataset(dataset_name, source_file) == "primary"


def load_primary_dataset_paths() -> list[Path]:
    """Return the configured primary dataset files as absolute paths."""
    selection = load_dataset_selection()
    return [normalize_config_path_to_absolute(path) for path in sorted(selection["primary_files"])]


def load_primary_dataset_names() -> list[str]:
    """Return the configured primary dataset names."""
    selection = load_dataset_selection()
    return sorted(selection["primary_datasets"])


def selected_baseline_flag(file_path: str | Path) -> bool:
    """Return whether a discovered file is in the final baseline set."""
    return classify_dataset_file(file_path)[3]


def build_inventory_selection_note(file_path: str | Path) -> str:
    """Compatibility helper for inventory note generation."""
    return selection_note(file_path)


def classify_dataset_path(file_path: str | Path) -> str:
    """Return only the dataset tier for a path."""
    return classify_dataset_file(file_path)[0]


def is_selected_primary_file(source_file: str | Path) -> bool:
    """Return True when the file is explicitly selected for the baseline set."""
    return _normalize_selection_path(source_file) in selected_primary_source_files()


def is_final_baseline_dataset(file_path: str | Path) -> bool:
    """Return whether a discovered file is in the final baseline set."""
    return selected_baseline_flag(file_path)


def primary_dataset_files_from_config() -> list[Path]:
    return load_primary_dataset_paths()


def primary_dataset_names_from_config() -> list[str]:
    return load_primary_dataset_names()


def load_dataset(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a dataset from disk based on its file extension."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path, **kwargs)
    if suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t", **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(file_path, **kwargs)
    if suffix == ".json":
        return pd.read_json(file_path, **kwargs)
    if suffix == ".arff":
        return load_arff_dataset(file_path)

    raise ValueError(f"Unsupported dataset format: {suffix}")


def load_arff_dataset(path: str | Path) -> pd.DataFrame:
    """Load an ARFF dataset if scipy is available."""
    try:
        from scipy.io import arff
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ARFF support requires scipy. Install scipy or convert the dataset to CSV first."
        ) from exc

    file_path = Path(path)
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    cleaned_lines: list[str] = []
    in_data_section = False

    for line in raw_text.splitlines():
        stripped = line.strip()
        if not in_data_section:
            cleaned_lines.append(line)
            if stripped.lower() == "@data":
                in_data_section = True
            continue

        if not stripped:
            cleaned_lines.append(line)
            continue

        if stripped.startswith("###") or stripped.startswith("%"):
            continue

        cleaned_lines.append(line)

    data, _ = arff.loadarff(StringIO("\n".join(cleaned_lines)))
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    return df


def discover_raw_dataset_files(raw_dir: str | Path) -> list[Path]:
    """Return supported dataset files found under the raw data directory."""
    raw_path = Path(raw_dir)
    return sorted(
        [path for path in raw_path.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES]
    )


def _compute_imbalance_ratio(num_defective: int, num_clean: int) -> float | None:
    if num_defective == 0 or num_clean == 0:
        return None
    majority = max(num_defective, num_clean)
    minority = min(num_defective, num_clean)
    return float(majority / minority)


def profile_dataset(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    dataset_name: str,
    source_file: str | Path,
    metrics_columns: list[str] | None = None,
    status: str = "ok",
    notes: str = "",
) -> dict[str, Any]:
    """Build a detailed profiling record for a dataset after cleaning."""
    metrics_columns = metrics_columns or []
    source_path = Path(source_file)
    dataset_tier = classify_dataset(dataset_name, source_path)
    is_primary = dataset_tier == "primary"
    is_supplementary = dataset_tier == "supplementary"

    num_modules = int(cleaned_df["module_id"].nunique()) if "module_id" in cleaned_df.columns else int(len(cleaned_df))
    if "label" in cleaned_df.columns:
        num_defective = int((cleaned_df["label"] == 1).sum())
        num_clean = int((cleaned_df["label"] == 0).sum())
    else:
        num_defective = 0
        num_clean = 0

    has_commit_text = "commit_text" in cleaned_df.columns and cleaned_df["commit_text"].astype(str).str.strip().ne("").any()
    has_project_name = "project_name" in cleaned_df.columns
    has_metrics = any(metric in cleaned_df.columns for metric in metrics_columns)
    defect_rate = float(num_defective / num_modules) if num_modules else None
    imbalance_ratio = _compute_imbalance_ratio(num_defective, num_clean)

    return {
        "dataset_name": dataset_name,
        "dataset_tier": dataset_tier,
        "is_primary": bool(is_primary),
        "is_supplementary": bool(is_supplementary),
        "source_file": str(source_path),
        "format": source_path.suffix.lower(),
        "num_rows_raw": int(len(raw_df)),
        "num_columns_raw": int(len(raw_df.columns)),
        "num_rows_clean": int(len(cleaned_df)),
        "num_columns_clean": int(len(cleaned_df.columns)),
        "num_modules": num_modules,
        "num_defective": num_defective,
        "num_clean": num_clean,
        "imbalance_ratio": imbalance_ratio,
        "defect_rate": defect_rate,
        "has_metrics": has_metrics,
        "has_commit_text": bool(has_commit_text),
        "has_project_name": bool(has_project_name),
        "status": status,
        "notes": notes,
    }


def build_dataset_inventory(raw_dir: str | Path) -> pd.DataFrame:
    """Build a lightweight inventory table for discovered dataset files."""
    records: list[dict[str, Any]] = []

    for file_path in discover_raw_dataset_files(raw_dir):
        record: dict[str, Any] = {
            "dataset_name": file_path.stem,
            "source_file": str(file_path),
            "format": file_path.suffix.lower(),
            "num_rows_raw": None,
            "num_columns_raw": None,
            "has_commit_text": False,
            "has_label": False,
            "status": "ok",
            "notes": "",
        }
        try:
            df = load_dataset(file_path)
            lowered_columns = {col.lower() for col in df.columns}
            record["num_rows_raw"] = len(df)
            record["num_columns_raw"] = len(df.columns)
            record["has_commit_text"] = any(col in lowered_columns for col in TEXT_COLUMN_CANDIDATES)
            record["has_label"] = any(col in lowered_columns for col in LABEL_COLUMN_CANDIDATES)
        except Exception as exc:  # pragma: no cover
            record["status"] = "error"
            record["notes"] = str(exc)
        records.append(record)

    return pd.DataFrame(records)


def prepare_dataset_from_raw(
    file_path: str | Path,
    dataset_name: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, normalize, validate, clean, and profile one raw dataset file."""
    resolved_path = Path(file_path)
    resolved_name = dataset_name or resolved_path.stem
    raw_df = load_dataset(resolved_path)
    unified_df = unify_schema(raw_df, dataset_name=resolved_name)
    validate_dataset_schema(unified_df)
    cleaned_df, clean_summary = clean_dataset(unified_df, return_summary=True)
    profile = profile_dataset(
        raw_df=raw_df,
        cleaned_df=cleaned_df,
        dataset_name=resolved_name,
        source_file=resolved_path,
        status="ok",
        notes="prepared from raw input",
    )
    profile["clean_summary"] = clean_summary
    return cleaned_df, profile
