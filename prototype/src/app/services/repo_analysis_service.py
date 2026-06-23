"""Repository analysis helpers for the Streamlit demo."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
import zipfile
from typing import Any

import pandas as pd

import joblib
import numpy as np

from src.app.services.dataset_service import DEFAULT_METRICS, MODULE_LEVEL_DATASETS
from src.app.services.evaluation_service import load_best_models_table, load_final_models_table, row_to_dict
from src.app.services.model_service import build_sample_predictions
from src.app.state import AnalysisResultRow, AnalysisResultState, StatusMessage
from src.utils.logging import get_logger
from src.utils.paths import MODELS_DIR
from src.utils.provenance import artifact_uses_commit_text

logger = get_logger(__name__)

MAX_TEXT_FILE_BYTES = 1_000_000
MAX_ARCHIVE_TOTAL_BYTES = 10_000_000
MAX_ARCHIVE_FILES = 500

TEXT_FILE_EXTENSIONS = {
    ".py", ".txt", ".md", ".json", ".csv", ".yaml", ".yml", ".ini", ".toml", ".cfg",
    ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".scss", ".sh", ".bat", ".ps1",
}
HIGH_RISK_PATTERNS = [
    (re.compile(r"\b(auth|token|session|login|password|jwt)\b", re.IGNORECASE), "auth"),
    (re.compile(r"\b(api|route|endpoint|controller|service)\b", re.IGNORECASE), "api"),
    (re.compile(r"\b(sql|query|db|database|orm|repository)\b", re.IGNORECASE), "data"),
    (re.compile(r"\b(validate|schema|sanitize|escape|guard)\b", re.IGNORECASE), "validation"),
    (re.compile(r"\b(test|pytest|unittest|assert)\b", re.IGNORECASE), "tests"),
]
DOCS_DIR_PATTERN = re.compile(r"(^|/)(docs?)(/|$)", re.IGNORECASE)
DOCS_BASENAME_PATTERN = re.compile(r"^(readme|changelog|changes|contributing|license|copying)([._-].*)?$", re.IGNORECASE)
DOCS_EXTENSIONS = {".md", ".rst", ".adoc", ".markdown"}
SOURCE_DIR_HINTS = ("src/", "app/", "api/", "service/", "services/", "core/", "lib/", "backend/", "models/", "modules/")
CONFIG_FILE_HINTS = ("config", "settings", "routes", "main", "server", "app")
LOW_PRIORITY_FILE_HINTS = ("sample", "example", "fixture", "mock", "demo", "temp", "tmp", "generated", "vendor", "dist", "build", "coverage")
CONTROL_FLOW_PATTERNS = (
    re.compile(r"\b(if|elif|else|for|while|try|except|with|switch|case|catch)\b", re.IGNORECASE),
    re.compile(r"\b(return|break|continue|raise|throw|yield)\b", re.IGNORECASE),
)
BUILD_ARTIFACT_PATTERNS = (
    re.compile(r"(^|/)node_modules(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)(dist|build|out|target|coverage|\.next|\.nuxt|\.svelte-kit)(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|composer\.lock|gemfile\.lock|cargo\.lock)$", re.IGNORECASE),
    re.compile(r"\.min\.(js|css|mjs|cjs)$", re.IGNORECASE),
    re.compile(r"\.bundle\.(js|css|mjs|cjs)$", re.IGNORECASE),
    re.compile(r"\.map$", re.IGNORECASE),
)

def _is_build_artifact(snapshot: "FileSnapshot") -> bool:
    path = snapshot.path.replace("\\", "/")
    return any(pattern.search(path) for pattern in BUILD_ARTIFACT_PATTERNS)


def _path_has_low_priority_hint(normalized_path: str) -> bool:
    segments = [segment for segment in normalized_path.replace("\\", "/").split("/") if segment]
    if len(segments) <= 1:
        return False
    candidate_segments = segments[1:]
    return any(
        hint == segment or segment.startswith(f"{hint}.") or segment.startswith(f"{hint}-") or segment.startswith(f"{hint}_")
        for hint in LOW_PRIORITY_FILE_HINTS
        for segment in candidate_segments
    )


@dataclass
class FileSnapshot:
    path: str
    name: str
    text: str
    line_count: int
    size: int
    extension: str
    is_binary: bool = False


@dataclass
class RiskRow:
    path: str
    probability: float
    severity: str
    reason: str
    signals: list[str]
    source_type: str = "heuristic"
    model_probability: str | None = None
    model_prediction: Any | None = None
    notes: list[str] | None = None


def _severity(probability: float) -> str:
    if probability >= 0.9:
        return "Critical"
    if probability >= 0.75:
        return "High"
    if probability >= 0.5:
        return "Medium"
    return "Low"


def _is_docs_file(snapshot: FileSnapshot) -> bool:
    path = snapshot.path.replace("\\", "/")
    name = snapshot.name.lower()
    return (
        DOCS_DIR_PATTERN.search(path) is not None
        or DOCS_BASENAME_PATTERN.match(name) is not None
        or snapshot.extension in DOCS_EXTENSIONS
    )


def _decode_text(name: str, raw: bytes, prefix: str) -> FileSnapshot:
    suffix = Path(name).suffix.lower()
    if suffix not in TEXT_FILE_EXTENSIONS:
        return FileSnapshot(path=f"{prefix}/{name}", name=name, text="", line_count=0, size=len(raw), extension=suffix, is_binary=True)
    if len(raw) > MAX_TEXT_FILE_BYTES:
        return FileSnapshot(path=f"{prefix}/{name}", name=name, text="", line_count=0, size=len(raw), extension=suffix, is_binary=True)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except UnicodeDecodeError:
            return FileSnapshot(path=f"{prefix}/{name}", name=name, text="", line_count=0, size=len(raw), extension=suffix, is_binary=True)
    lines = [line for line in text.splitlines() if line.strip()]
    return FileSnapshot(path=f"{prefix}/{name}", name=name, text=text, line_count=len(lines), size=len(raw), extension=suffix, is_binary=False)


def _safe_github_archive_relative_path(member_path: str) -> str | None:
    """Return a normalized repository-relative path for a GitHub archive member."""
    normalized = member_path.replace("\\", "/")
    archive_path = PurePosixPath(normalized)
    if archive_path.is_absolute():
        return None

    parts = [part for part in archive_path.parts if part and part != "."]
    if len(parts) < 2 or any(part == ".." for part in parts):
        return None

    relative_path = PurePosixPath(*parts[1:])
    if not relative_path.parts or any(part in {"", ".", ".."} for part in relative_path.parts):
        return None
    return str(relative_path)


def _extract_github_owner_repo(repo_url: str) -> tuple[str, str] | None:
    parsed = urllib.parse.urlparse(repo_url.strip())
    if parsed.scheme != "https":
        return None
    hostname = (parsed.hostname or "").lower()
    if hostname not in {"github.com", "www.github.com"}:
        return None
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        return None
    owner = parts[0]
    repo = parts[1].removesuffix(".git")
    safe_segment = re.compile(r"^[A-Za-z0-9_.-]+$")
    if not safe_segment.match(owner) or not safe_segment.match(repo):
        return None
    return owner, repo


def _download_github_zip(repo_url: str) -> tuple[list[FileSnapshot], list[str], str, list[str]]:
    owner_repo = _extract_github_owner_repo(repo_url)
    if not owner_repo:
        return [], ["Only HTTPS GitHub repository URLs are supported for direct download."], repo_url, []

    owner, repo = owner_repo
    source_label = f"{owner}/{repo}"
    zip_url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/main"
    notes = [f"Attempting to download repository archive: {zip_url}"]
    excluded_files: list[str] = []

    try:
        with urllib.request.urlopen(zip_url, timeout=30) as response:
            raw = response.read()
    except Exception as exc:
        logger.debug("Failed to download GitHub main archive %s: %s", zip_url, exc)
        zip_url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/master"
        notes.append("Main branch archive failed; retrying master.")
        try:
            with urllib.request.urlopen(zip_url, timeout=30) as response:
                raw = response.read()
        except Exception as exc:
            return [], [f"Failed to download repository archive: {exc}"], source_label, []

    snapshots: list[FileSnapshot] = []
    try:
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            analyzed_files = 0
            total_uncompressed = 0
            unsafe_paths = 0
            for member in archive.infolist():
                if member.is_dir():
                    continue
                if analyzed_files >= MAX_ARCHIVE_FILES:
                    notes.append(f"Stopped after {MAX_ARCHIVE_FILES} supported files to keep analysis bounded.")
                    break
                relative_path = _safe_github_archive_relative_path(member.filename)
                if relative_path is None:
                    unsafe_paths += 1
                    continue
                if not relative_path or relative_path.startswith("."):
                    continue
                suffix = Path(relative_path).suffix.lower()
                if suffix not in TEXT_FILE_EXTENSIONS:
                    continue
                total_uncompressed += int(member.file_size)
                if member.file_size > MAX_TEXT_FILE_BYTES:
                    notes.append(f"Skipped oversized repository file: {relative_path}")
                    continue
                if total_uncompressed > MAX_ARCHIVE_TOTAL_BYTES:
                    notes.append(f"Stopped repository extraction after {MAX_ARCHIVE_TOTAL_BYTES} uncompressed bytes.")
                    break
                try:
                    data = archive.read(member)
                except Exception as exc:
                    logger.debug("Failed to read repository archive member %s: %s", relative_path, exc)
                    continue
                snapshot = _decode_text(Path(relative_path).name, data, prefix=source_label)
                snapshot.path = relative_path
                if _is_docs_file(snapshot):
                    excluded_files.append(relative_path)
                    continue
                snapshots.append(snapshot)
                analyzed_files += 1
            if unsafe_paths:
                notes.append(f"Skipped {unsafe_paths} unsafe repository archive path(s).")
    except zipfile.BadZipFile as exc:
        return [], [f"Downloaded archive could not be read: {exc}"], source_label, []

    if excluded_files:
        notes.append(f"Excluded {len(excluded_files)} documentation file(s) from repository analysis.")
    if not snapshots:
        notes.append("Repository archive contained no supported text files.")
    return snapshots, notes, source_label, excluded_files


def _build_feature_row(snapshot: FileSnapshot) -> dict[str, Any]:
    normalized_path = snapshot.path.replace("\\", "/").lower()
    normalized_name = snapshot.name.lower()
    text = snapshot.text or ""
    pattern_hits = {tag for pattern, tag in HIGH_RISK_PATTERNS if pattern.search(text)}
    control_flow_count = sum(len(pattern.findall(text)) for pattern in CONTROL_FLOW_PATTERNS)

    loc = max(snapshot.line_count, 1)
    vg = max(1, control_flow_count + len(pattern_hits))
    evg = max(1, vg + (1 if any(hint in normalized_path for hint in SOURCE_DIR_HINTS) else 0))
    ivg = max(1, len(pattern_hits) + (1 if any(hint in normalized_name for hint in CONFIG_FILE_HINTS) else 0))
    branch_count = max(1, control_flow_count)

    penalty = 1.0
    if _path_has_low_priority_hint(normalized_path):
        penalty *= 0.55
    if "/tests/" in normalized_path or normalized_path.startswith("tests/"):
        penalty *= 0.72

    return {
        "module_id": normalized_path or snapshot.name,
        "label": 0,
        "loc": float(loc),
        "v(g)": float(vg),
        "ev(g)": float(evg),
        "iv(g)": float(ivg),
        "branchCount": float(branch_count),
        "_pattern_hits": pattern_hits,
        "_path_penalty": penalty,
    }


def _score_snapshot(snapshot: FileSnapshot, *, extension_weight: float = 1.0, directory_weight: float = 1.0) -> RiskRow:
    feature_row = _build_feature_row(snapshot)
    normalized_path = snapshot.path.replace("\\", "/").lower()
    normalized_name = snapshot.name.lower()
    pattern_hits = set(feature_row.pop("_pattern_hits", set()))
    path_penalty = float(feature_row.pop("_path_penalty", 1.0))

    probability = 0.18
    probability += min(feature_row["loc"] / 800.0, 0.12)
    probability += min(feature_row["v(g)"] / 60.0, 0.18)
    probability += min(feature_row["ev(g)"] / 60.0, 0.08)
    probability += min(feature_row["iv(g)"] / 30.0, 0.04)
    probability += min(feature_row["branchCount"] / 60.0, 0.06)
    probability += 0.025 * len(pattern_hits)

    if snapshot.extension in {".py", ".js", ".ts", ".tsx", ".jsx"}:
        probability += 0.04
    if snapshot.extension in {".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}:
        probability -= 0.05
    if any(hint in normalized_path for hint in SOURCE_DIR_HINTS):
        probability += 0.04
    if any(hint in normalized_name for hint in CONFIG_FILE_HINTS):
        probability += 0.02
    if _path_has_low_priority_hint(normalized_path):
        probability -= 0.18
    if "/tests/" in normalized_path or normalized_path.startswith("tests/"):
        probability -= 0.12
    if snapshot.line_count == 0:
        probability = min(probability, 0.30)
    if len(snapshot.text) > 10000:
        probability += 0.04
    if _is_build_artifact(snapshot):
        probability = min(probability, 0.20)
    probability *= path_penalty
    probability += 0.01 * (extension_weight - 1.0)
    probability += 0.01 * (directory_weight - 1.0)
    probability = max(0.05, min(probability, 0.92))

    reason_bits: list[str] = []
    if snapshot.line_count > 120 or len(snapshot.text) > 4000:
        reason_bits.append("large surface area")
    if pattern_hits:
        reason_bits.append(", ".join(sorted(pattern_hits)))
    if any(hint in normalized_path for hint in SOURCE_DIR_HINTS):
        reason_bits.append("core source path")
    if any(hint in normalized_name for hint in CONFIG_FILE_HINTS):
        reason_bits.append("core config/entrypoint hint")
    if _path_has_low_priority_hint(normalized_path):
        reason_bits.append("lower-priority sample/example path")
    if not reason_bits:
        reason_bits.append("noisy or under-specified code surface")

    signals = [
        f"{snapshot.line_count} logical lines",
        f"size={snapshot.size} bytes",
        f"extension={snapshot.extension or 'unknown'}",
        f"loc={feature_row['loc']:.0f}",
        f"v(g)={feature_row['v(g)']:.0f}",
        f"ev(g)={feature_row['ev(g)']:.0f}",
        f"iv(g)={feature_row['iv(g)']:.0f}",
        f"branchCount={feature_row['branchCount']:.0f}",
    ]
    if snapshot.is_binary:
        signals.append("binary or unsupported content")
    else:
        signals.extend(sorted(pattern_hits) or ["no high-risk keywords detected"])
        if any(hint in normalized_path for hint in SOURCE_DIR_HINTS):
            signals.append("core source path")
        if any(hint in normalized_name for hint in CONFIG_FILE_HINTS):
            signals.append("entrypoint/config hint")
        if _path_has_low_priority_hint(normalized_path):
            signals.append("sample/example penalty")
        if "/tests/" in normalized_path or normalized_path.startswith("tests/"):
            signals.append("test-path penalty")

    return RiskRow(
        path=snapshot.path,
        probability=probability,
        severity=_severity(probability),
        reason=f"Detected {', '.join(reason_bits)}.",
        signals=signals,
        source_type="heuristic",
        notes=["This score was derived from lightweight repo heuristics."],
    )


def _score_project(snapshots: list[FileSnapshot], notes: list[str]) -> list[RiskRow]:
    rows: list[RiskRow] = []
    docs_filtered = 0
    total_files = len(snapshots) or 1
    source_files = sum(1 for snapshot in snapshots if any(hint in snapshot.path.replace("\\", "/").lower() for hint in SOURCE_DIR_HINTS))
    config_files = sum(1 for snapshot in snapshots if any(hint in snapshot.name.lower() for hint in CONFIG_FILE_HINTS))
    source_density = source_files / total_files
    config_density = config_files / total_files

    for snapshot in snapshots:
        if _is_docs_file(snapshot):
            docs_filtered += 1
            continue
        if _is_build_artifact(snapshot):
            continue
        extension = snapshot.extension.lower()
        extension_weight = 1.0
        directory_weight = 1.0
        if extension in {".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".ps1"}:
            extension_weight += 0.08 + (0.10 * source_density)
        elif extension in {".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}:
            extension_weight += 0.04 + (0.06 * config_density)
        elif extension in {".md", ".rst", ".adoc", ".markdown"}:
            extension_weight -= 0.12
        if any(hint in snapshot.path.replace("\\", "/").lower() for hint in SOURCE_DIR_HINTS):
            directory_weight += 0.10 + (0.20 * source_density)
        if snapshot.is_binary:
            rows.append(
                RiskRow(
                    snapshot.path,
                    0.52,
                    _severity(0.52),
                    "Binary or unsupported file should be reviewed manually.",
                    ["Binary or unsupported content", f"size={snapshot.size} bytes"],
                    source_type="heuristic",
                    notes=["This file could not be analyzed deeply because it is binary or unsupported."],
                )
            )
        else:
            rows.append(_score_snapshot(snapshot, extension_weight=extension_weight, directory_weight=directory_weight))

    if docs_filtered:
        notes.append("Documentation files were excluded from default risk ranking.")

    deduped: dict[str, RiskRow] = {}
    for row in rows:
        current = deduped.get(row.path)
        if current is None or row.probability > current.probability:
            deduped[row.path] = row

    return sorted(deduped.values(), key=lambda item: item.probability, reverse=True)[:15]


HYBRID_MODELS_DIR = MODELS_DIR / "hybrid_tfidf"


def _extract_git_commit_messages(repo_url: str, snapshots: list["FileSnapshot"]) -> dict[str, str]:
    """Clone the repository shallowly and extract the latest subject per file."""
    owner_repo = _extract_github_owner_repo(repo_url)
    if not owner_repo:
        return {}

    owner, repo = owner_repo
    clone_url = f"https://github.com/{owner}/{repo}.git"
    wanted_paths = {snapshot.path.replace("\\", "/") for snapshot in snapshots}
    commit_map: dict[str, str] = {}

    tmp_dir = tempfile.mkdtemp(prefix="sdp_git_")
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "50", "--single-branch", clone_url, tmp_dir],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return {}

        log_result = subprocess.run(
            ["git", "log", "--name-only", "--format=%x1e%s", "--"],
            capture_output=True, text=True, timeout=30, cwd=tmp_dir,
        )
        if log_result.returncode != 0 or not log_result.stdout.strip():
            return {}

        current_subject = ""
        for raw_line in log_result.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("\x1e"):
                current_subject = line[1:].strip()
                continue
            file_path = line.replace("\\", "/")
            if current_subject and file_path in wanted_paths and file_path not in commit_map:
                commit_map[file_path] = current_subject
                if len(commit_map) == len(wanted_paths):
                    break
    except Exception as exc:
        logger.debug("Failed to clone repository for commit-message extraction: %s", exc)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return commit_map


def _hybrid_model_candidates_from_final_selection() -> list[tuple[Path, str]]:
    """Return final selected hybrid model paths in selection order."""
    final_df = load_final_models_table()
    if final_df.empty or "model_path" not in final_df.columns:
        return []

    rows = []
    for _, row in final_df.iterrows():
        row_dict = row_to_dict(row)
        if not artifact_uses_commit_text(row_dict):
            continue
        model_path = str(row_dict.get("model_path", "")).strip()
        if not model_path:
            continue
        rows.append(row_dict)

    def sort_key(row: dict[str, Any]) -> tuple[float, float, str]:
        selection_rank = pd.to_numeric(pd.Series([row.get("selection_rank")]), errors="coerce").iloc[0]
        rank_within_dataset = pd.to_numeric(pd.Series([row.get("rank_within_dataset")]), errors="coerce").iloc[0]
        return (
            float(selection_rank) if pd.notna(selection_rank) else float("inf"),
            float(rank_within_dataset) if pd.notna(rank_within_dataset) else float("inf"),
            str(row.get("dataset_name", "")),
        )

    candidates: list[tuple[Path, str]] = []
    for row in sorted(rows, key=sort_key):
        path = Path(str(row.get("model_path", "")).strip())
        label = f"{row.get('dataset_name', path.stem)}/{row.get('model', path.stem)}"
        candidates.append((path, label))
    return candidates


def _load_hybrid_model() -> tuple[Any | None, str | None]:
    """Load the final selected hybrid model bundle for live inference."""
    for path, label in _hybrid_model_candidates_from_final_selection():
        if path.exists():
            try:
                bundle = joblib.load(path)
                if hasattr(bundle, "preprocessor") and bundle.preprocessor is not None:
                    return bundle, label
            except Exception as exc:
                logger.debug("Failed to load hybrid model candidate %s: %s", path, exc)
                continue
    return None, None


def _predict_with_hybrid_model(snapshots: list["FileSnapshot"], commit_messages: dict[str, str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Use the hybrid metrics+TF-IDF model for commit-aware predictions."""
    notes: list[str] = []
    bundle, model_name = _load_hybrid_model()
    if bundle is None:
        return {}, ["No hybrid model available; commit message features not used."]

    spec = bundle.preprocessor
    metrics_list = spec.metrics_spec.selected_metrics

    rows = []
    for snapshot in snapshots:
        if _is_docs_file(snapshot) or _is_build_artifact(snapshot):
            continue
        feature_row = _build_feature_row(snapshot)
        file_path = snapshot.path.replace("\\", "/")
        # Map heuristic metrics to the model's expected metric names
        row_data = {"module_id": file_path, "label": 0}
        # Use commit messages from git log as commit_text
        row_data["commit_text"] = commit_messages.get(file_path, "")
        # Map our computed metrics
        metric_mapping = {
            "loc": feature_row.get("loc", 0),
            "v(g)": feature_row.get("v(g)", 0),
            "ev(g)": feature_row.get("ev(g)", 0),
            "iv(g)": feature_row.get("iv(g)", 0),
            "branchCount": feature_row.get("branchCount", 0),
        }
        for m in metrics_list:
            row_data[m] = metric_mapping.get(m, 0.0)
        rows.append(row_data)

    if not rows:
        return {}, ["No analyzable files for hybrid model."]

    df = pd.DataFrame(rows)
    has_text = df["commit_text"].str.strip().ne("").sum()

    try:
        X = spec.transform(df)
        proba = bundle.predict_proba(X)
        if hasattr(proba, 'shape') and proba.ndim == 2:
            pos_proba = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        else:
            pos_proba = np.array(proba)

        prediction_map: dict[str, dict[str, Any]] = {}
        decision_threshold = getattr(bundle, "decision_threshold", 0.5) or 0.5
        for i, row_data in enumerate(rows):
            module_id = row_data["module_id"]
            prediction_map[module_id] = {
                "module_id": module_id,
                "probability": float(pos_proba[i]),
                "prediction": int(pos_proba[i] >= decision_threshold),
                "commit_text_used": bool(row_data.get("commit_text", "").strip()),
                "commit_text_preview": (row_data.get("commit_text", "")[:80] + "...") if len(row_data.get("commit_text", "")) > 80 else row_data.get("commit_text", ""),
            }

        notes.append(f"Hybrid model ({model_name}) used: metrics + commit-message TF-IDF features.")
        notes.append(f"Commit messages found for {has_text}/{len(rows)} files.")
        return prediction_map, notes
    except Exception as exc:
        logger.debug("Hybrid model inference failed: %s", exc)
        return {}, [f"Hybrid model inference failed: {exc}"]


def _predict_with_model_if_possible(snapshots: list[FileSnapshot]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    notes: list[str] = []
    best_df = load_best_models_table()
    if best_df.empty:
        return {}, ["No model table available for live inference; using heuristic scoring."]

    module_level_df = best_df[best_df["dataset_name"].astype(str).isin(MODULE_LEVEL_DATASETS)] if "dataset_name" in best_df.columns else best_df
    if module_level_df.empty:
        return {}, ["No module-level model artifact available for file-level inference; using heuristic scoring."]

    best_row = row_to_dict(module_level_df.iloc[0])
    model_path = best_row.get("model_path")
    if not model_path or not Path(model_path).exists():
        return {}, ["No valid model artifact available for live inference; using heuristic scoring."]

    feature_rows = [_build_feature_row(snapshot) for snapshot in snapshots if not _is_docs_file(snapshot)]
    if not feature_rows:
        return {}, ["No analyzable files available for model inference."]

    feature_df = pd.DataFrame(feature_rows)
    for metric in DEFAULT_METRICS:
        if metric not in feature_df.columns:
            feature_df[metric] = 0.0

    prediction_input = feature_df[[*DEFAULT_METRICS]].copy()
    prediction_input["module_id"] = feature_df["module_id"]
    prediction_input["label"] = feature_df.get("label", 0)

    selected_row = {**best_row, "model_path": model_path}
    try:
        predictions, status = build_sample_predictions(selected_row, prediction_input, list(DEFAULT_METRICS))
    except Exception as exc:
        logger.debug("Model inference failed for live repository analysis: %s", exc)
        return {}, [f"Model inference failed; using heuristic scoring. ({exc})"]

    if not status.available or not predictions:
        return {}, [f"Model inference unavailable: {status.message}; using heuristic scoring."]

    prediction_map: dict[str, dict[str, Any]] = {}
    for row in predictions:
        module_id = str(row.get("module_id", ""))
        prediction_map[module_id] = row
    notes.append(f"Model-backed inference used artifact: {model_path}")
    return prediction_map, notes


def _apply_model_predictions(rows: list[RiskRow], prediction_map: dict[str, dict[str, Any]]) -> None:
    if not prediction_map:
        return

    for row in rows:
        module_id = row.path.replace("\\", "/").lower()
        candidate_keys = [module_id, module_id.split(":")[-1]]
        predicted = None
        for key in candidate_keys:
            if key in prediction_map:
                predicted = prediction_map[key]
                break
        if predicted:
            row.source_type = "model+heuristic"
            row.model_prediction = predicted.get("prediction")
            if "probability" in predicted:
                row.model_probability = f"{float(predicted['probability']) * 100:.0f}%"
                row.notes = (row.notes or []) + ["Model-backed probability is available for this file."]


def build_analysis_result(repo_url: str) -> AnalysisResultState:
    snapshots: list[FileSnapshot] = []
    notes: list[str] = []
    excluded_files: list[str] = []

    source_value = repo_url.strip()
    repo_label = source_value or "none"
    if repo_url.strip():
        repo_snapshots, repo_notes, repo_label, repo_excluded = _download_github_zip(repo_url)
        snapshots.extend(repo_snapshots)
        notes.extend(repo_notes)
        excluded_files.extend(repo_excluded)
        notes.append(f"Repository source: {repo_label}")
    else:
        notes.append("No repository URL was provided.")

    if not snapshots:
        return AnalysisResultState(
            source=repo_label,
            file_count=0,
            risks=[],
            notes=[*notes, "No analyzable input was provided."],
            excluded_files=excluded_files,
            explainability=StatusMessage(available=False, message="No analyzable files available for scoring.", details={}),
        )

    prediction_map, prediction_notes = _predict_with_model_if_possible(snapshots)
    notes.extend(prediction_notes)

    # Attempt hybrid commit-message-aware prediction for repo URLs
    hybrid_map: dict[str, dict[str, Any]] = {}
    if repo_url.strip():
        commit_messages = _extract_git_commit_messages(repo_url, snapshots)
        if commit_messages:
            hybrid_map, hybrid_notes = _predict_with_hybrid_model(snapshots, commit_messages)
            notes.extend(hybrid_notes)
        else:
            notes.append("Could not extract commit messages from repository; using metrics-only model.")

    # Prefer hybrid predictions over metrics-only when available
    effective_map = hybrid_map if hybrid_map else prediction_map

    risk_rows = _score_project(snapshots, notes)
    _apply_model_predictions(risk_rows, effective_map)

    # Add commit-text signals to risk rows when hybrid was used
    if hybrid_map:
        for row in risk_rows:
            module_id = row.path.replace("\\", "/").lower()
            pred = hybrid_map.get(module_id) or hybrid_map.get(row.path)
            if pred and pred.get("commit_text_used"):
                row.signals = row.signals or []
                row.signals.append(f"commit_msg: \"{pred.get('commit_text_preview', '')}\"")
                row.source_type = "hybrid(metrics+commit_text)"

    explainability_status = StatusMessage(
        available=bool(effective_map),
        message="Hybrid model (metrics + commit-message TF-IDF) explanations available." if hybrid_map else (
            "Model-backed explanations are partially available." if prediction_map else "Using heuristic explanations only."
        ),
        details={
            "model_predictions_available": bool(effective_map),
            "hybrid_model_used": bool(hybrid_map),
            "excluded_file_count": len(excluded_files),
        },
    )

    return AnalysisResultState(
        source=repo_label,
        file_count=len(snapshots),
        risks=[
            AnalysisResultRow(
                path=row.path,
                probability=f"{row.probability * 100:.0f}%",
                severity=row.severity,
                reason=row.reason,
                signals=row.signals,
                source_type=row.source_type,
                model_probability=row.model_probability,
                model_prediction=row.model_prediction,
                notes=row.notes or [],
            )
            for row in risk_rows
        ],
        notes=notes,
        excluded_files=excluded_files,
        explainability=explainability_status,
    )


__all__ = [
    "AnalysisResultRow",
    "AnalysisResultState",
    "build_analysis_result",
]
