"""Repository and upload analysis helpers for the Streamlit demo."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import hashlib
import re
import urllib.parse
import urllib.request
import zipfile
from typing import Any

import pandas as pd

from src.app.services.dataset_service import DEFAULT_METRICS
from src.app.services.evaluation_service import load_best_models_table, row_to_dict
from src.app.services.model_service import build_sample_predictions
from src.app.state import AnalysisResultRow, AnalysisResultState, StatusMessage

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
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except UnicodeDecodeError:
            return FileSnapshot(path=f"{prefix}/{name}", name=name, text="", line_count=0, size=len(raw), extension=suffix, is_binary=True)
    lines = [line for line in text.splitlines() if line.strip()]
    return FileSnapshot(path=f"{prefix}/{name}", name=name, text=text, line_count=len(lines), size=len(raw), extension=suffix, is_binary=False)


def _extract_zip(name: str, raw: bytes) -> tuple[list[FileSnapshot], list[str], list[str]]:
    snapshots: list[FileSnapshot] = []
    notes: list[str] = []
    excluded_files: list[str] = []
    try:
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                member_path = member.filename
                suffix = Path(member_path).suffix.lower()
                if suffix not in TEXT_FILE_EXTENSIONS:
                    continue
                try:
                    data = archive.read(member)
                except Exception:
                    continue
                snapshot = _decode_text(member_path, data, prefix=name)
                snapshot.path = f"{name}:{member_path}"
                if _is_docs_file(snapshot):
                    excluded_files.append(snapshot.path)
                    continue
                snapshots.append(snapshot)
    except zipfile.BadZipFile:
        notes.append("Uploaded archive is not a valid zip file.")
    if excluded_files:
        notes.append(f"Excluded {len(excluded_files)} documentation file(s) from uploaded archive analysis.")
    return snapshots, notes, excluded_files


def _extract_github_owner_repo(repo_url: str) -> tuple[str, str] | None:
    parsed = urllib.parse.urlparse(repo_url.strip())
    if "github.com" not in parsed.netloc.lower():
        return None
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        return None
    return parts[0], parts[1].removesuffix(".git")


def _download_github_zip(repo_url: str) -> tuple[list[FileSnapshot], list[str], str, list[str]]:
    owner_repo = _extract_github_owner_repo(repo_url)
    if not owner_repo:
        return [], ["Only GitHub repository URLs are supported for direct download."], repo_url, []

    owner, repo = owner_repo
    source_label = f"{owner}/{repo}"
    zip_url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/main"
    notes = [f"Attempting to download repository archive: {zip_url}"]
    excluded_files: list[str] = []

    try:
        with urllib.request.urlopen(zip_url, timeout=30) as response:
            raw = response.read()
    except Exception:
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
            for member in archive.infolist():
                if member.is_dir():
                    continue
                member_path = member.filename
                if "/" not in member_path:
                    continue
                relative_path = "/".join(member_path.split("/")[1:])
                if not relative_path or relative_path.startswith("."):
                    continue
                suffix = Path(relative_path).suffix.lower()
                if suffix not in TEXT_FILE_EXTENSIONS:
                    continue
                try:
                    data = archive.read(member)
                except Exception:
                    continue
                snapshot = _decode_text(Path(relative_path).name, data, prefix=source_label)
                snapshot.path = relative_path
                if _is_docs_file(snapshot):
                    excluded_files.append(relative_path)
                    continue
                snapshots.append(snapshot)
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

    probability = 0.32
    probability += min(feature_row["loc"] / 500.0, 0.16)
    probability += min(feature_row["v(g)"] / 20.0, 0.14)
    probability += min(feature_row["ev(g)"] / 20.0, 0.08)
    probability += min(feature_row["iv(g)"] / 20.0, 0.06)
    probability += min(feature_row["branchCount"] / 20.0, 0.05)
    probability += 0.07 * len(pattern_hits)

    if snapshot.extension in {".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".ps1"}:
        probability += 0.08
    if snapshot.extension in {".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"}:
        probability += 0.06
    if any(hint in normalized_path for hint in SOURCE_DIR_HINTS):
        probability += 0.08
    if any(hint in normalized_name for hint in CONFIG_FILE_HINTS):
        probability += 0.05
    if _path_has_low_priority_hint(normalized_path):
        probability -= 0.20
    if "/tests/" in normalized_path or normalized_path.startswith("tests/"):
        probability -= 0.14
    if snapshot.line_count == 0:
        probability = max(probability, 0.58)
    if len(snapshot.text) > 10000:
        probability += 0.06
    probability *= path_penalty
    probability = (probability * extension_weight) + (0.02 * directory_weight)
    probability = max(0.05, min(probability, 0.98))

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

    return sorted(deduped.values(), key=lambda item: item.probability, reverse=True)[:10]


def _predict_with_model_if_possible(snapshots: list[FileSnapshot]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    notes: list[str] = []
    best_df = load_best_models_table()
    if best_df.empty:
        return {}, ["No model table available for live inference; using heuristic scoring."]

    best_row = row_to_dict(best_df.iloc[0])
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
        candidate_keys = [module_id, f"pasted/{module_id.split('/')[-1]}", module_id.split(":")[-1]]
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


def _project_from_repo_url(repo_url: str) -> ProjectSource:
    snapshots, notes, label, _ = _download_github_zip(repo_url)
    return ProjectSource(source_type="repo", display_name=label, snapshots=snapshots, notes=notes)


def _project_from_upload(uploaded_file) -> tuple[ProjectSource, list[str]]:
    if uploaded_file is None:
        return ProjectSource(source_type="upload", display_name="", snapshots=[], notes=[]), []
    raw = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".zip"):
        snapshots, notes, excluded_files = _extract_zip(uploaded_file.name, raw)
        return ProjectSource(source_type="upload", display_name=uploaded_file.name, snapshots=snapshots, notes=notes), excluded_files
    snapshot = _decode_text(uploaded_file.name, raw, prefix="uploads")
    if _is_docs_file(snapshot):
        return ProjectSource(source_type="upload", display_name=uploaded_file.name, snapshots=[], notes=["Uploaded documentation file was excluded from default risk ranking."]), [snapshot.path]
    return ProjectSource(source_type="upload", display_name=uploaded_file.name, snapshots=[snapshot], notes=[]), []


def build_analysis_result(source_text: str, repo_url: str, uploaded_file) -> AnalysisResultState:
    snapshots: list[FileSnapshot] = []
    notes: list[str] = []
    excluded_files: list[str] = []

    if source_text.strip():
        code_hash = hashlib.sha1(source_text.strip().encode("utf-8")).hexdigest()[:8]
        snapshot = _decode_text(f"input_{code_hash}.py", source_text.strip().encode("utf-8"), prefix="pasted")
        snapshot.path = f"pasted/{snapshot.name}"
        if _is_docs_file(snapshot):
            excluded_files.append(snapshot.path)
            notes.append("Pasted documentation-like input was excluded from default risk ranking.")
        else:
            snapshots.append(snapshot)
            notes.append("Analyzed pasted code input.")

    if repo_url.strip():
        repo_snapshots, repo_notes, repo_label, repo_excluded = _download_github_zip(repo_url)
        snapshots.extend(repo_snapshots)
        notes.extend(repo_notes)
        excluded_files.extend(repo_excluded)
        notes.append(f"Repository source: {repo_label}")

    if uploaded_file is not None:
        upload_project, upload_excluded = _project_from_upload(uploaded_file)
        snapshots.extend(upload_project.snapshots)
        notes.extend(upload_project.notes)
        excluded_files.extend(upload_excluded)
        notes.append(f"Upload source: {upload_project.display_name}")

    if not snapshots:
        return AnalysisResultState(
            source="none",
            file_count=0,
            risks=[],
            notes=["No analyzable input was provided."],
            excluded_files=excluded_files,
            explainability=StatusMessage(available=False, message="No analyzable files available for scoring.", details={}),
        )

    prediction_map, prediction_notes = _predict_with_model_if_possible(snapshots)
    notes.extend(prediction_notes)

    risk_rows = _score_project(snapshots, notes)
    _apply_model_predictions(risk_rows, prediction_map)

    explainability_status = StatusMessage(
        available=bool(prediction_map),
        message="Model-backed explanations are partially available." if prediction_map else "Using heuristic explanations only.",
        details={
            "model_predictions_available": bool(prediction_map),
            "excluded_file_count": len(excluded_files),
        },
    )

    source_value = repo_url.strip() or (uploaded_file.name if uploaded_file else "pasted code")
    return AnalysisResultState(
        source=source_value,
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
    "AnalysisResultState",
    "AnalysisResultRow",
    "AnalysisResultState",
    "build_analysis_result",
]
