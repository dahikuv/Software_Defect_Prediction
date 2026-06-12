"""Streamlit view layer for the defect prediction MVC demo."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from src.app.controllers import DatasetDashboardState, build_dashboard_state, list_available_datasets
from src.app.services.repo_analysis_service import build_analysis_result

st.set_page_config(page_title="Defect Risk Analyzer", layout="wide")


@st.cache_data(show_spinner=False)
def get_available_datasets() -> list[str]:
    return list_available_datasets()


@st.cache_data(show_spinner=False)
def get_dashboard_state(dataset_name: str, selected_model: str | None) -> DatasetDashboardState:
    return build_dashboard_state(dataset_name, selected_model)


def _append_recent_analysis(analysis_result: dict[str, Any]) -> None:
    history = st.session_state.setdefault("recent_analyses", [])
    risks = analysis_result.get("risks", [])
    top = risks[0] if risks else {}
    history.insert(
        0,
        {
            "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": analysis_result.get("source", "N/A"),
            "files_analyzed": analysis_result.get("file_count", 0),
            "top_file": top.get("path", "N/A"),
            "top_risk": top.get("probability", "0%"),
            "scoring_source": top.get("source_type", "heuristic"),
        },
    )
    st.session_state["recent_analyses"] = history[:10]


def render_hero() -> None:
    st.title("Defect Risk Analyzer")
    st.write("Dán code, tải file, hoặc nhập link dự án để tìm file rủi ro cao.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mục tiêu", "High-risk files")
    c2.metric("Đầu vào", "Code / Repo / Upload")
    c3.metric("Kết quả", "Risk-ranked list")
    st.caption("Luồng chính: input → analyze → results.")


def render_input_card() -> tuple[str, str, Any, bool]:
    st.subheader("Analysis input")
    with st.form("analysis_form", clear_on_submit=False):
        col1, col2 = st.columns([2, 1])
        with col1:
            source_text = st.text_area(
                "Paste code here",
                placeholder="Paste your code or project notes here...",
                height=220,
                key="source_text_input",
            )
            project_link = st.text_input(
                "Project / repository link",
                placeholder="https://github.com/your/project",
                key="project_link_input",
            )
        with col2:
            uploaded = st.file_uploader("Upload file", type=["py", "txt", "md", "json", "csv", "zip"])
            st.caption("Paste code, add a link, or upload a file/zip.")
            analyze = st.form_submit_button("Analyze project", use_container_width=True)
    return source_text, project_link, uploaded, analyze


def render_status_message() -> None:
    if st.session_state.get("analysis_requested"):
        st.success("Phân tích xong. Kéo xuống để xem file rủi ro cao.")


def render_dashboard_summary(analysis_result: dict[str, Any] | None) -> None:
    st.subheader("Tổng quan")
    cards = st.columns(4)
    file_count = analysis_result.get("file_count", 0) if analysis_result else 0
    risk_count = len(analysis_result.get("risks", [])) if analysis_result else 0
    critical_count = 0
    if analysis_result:
        critical_count = sum(1 for row in analysis_result.get("risks", []) if row.get("severity") == "Critical")
    top_risk = analysis_result.get("risks", [{}])[0].get("probability", "0%") if analysis_result and analysis_result.get("risks") else "0%"

    cards[0].metric("Files analyzed", str(file_count))
    cards[1].metric("High-risk files", str(risk_count))
    cards[2].metric("Critical files", str(critical_count))
    cards[3].metric("Top risk", top_risk)

    left, right = st.columns(2)
    with left:
        st.markdown("**Risk trends**")
        if analysis_result and analysis_result.get("risks"):
            chart_values = [min(100, int(str(row["probability"]).rstrip("%"))) for row in analysis_result["risks"][:6]]
            st.line_chart(pd.DataFrame({"risk": chart_values}))
        else:
            st.line_chart(pd.DataFrame({"risk": [0, 0, 0, 0, 0, 0]}))
    with right:
        st.markdown("**Defect distribution**")
        if analysis_result and analysis_result.get("risks"):
            severity_counts = pd.Series([row["severity"] for row in analysis_result["risks"]]).value_counts()
            severity_df = pd.DataFrame({"count": severity_counts}).reindex(["Critical", "High", "Medium", "Low"], fill_value=0)
            st.bar_chart(severity_df)
        else:
            st.bar_chart(pd.DataFrame({"count": [0, 0, 0, 0]}, index=["Critical", "High", "Medium", "Low"]))


def render_recent_analyses() -> None:
    st.subheader("Lịch sử gần đây")
    history = st.session_state.get("recent_analyses", [])
    if not history:
        st.info("Chưa có lịch sử phân tích trong phiên này.")
        return
    st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


def render_results(analysis_result: dict[str, Any] | None) -> None:
    st.subheader("Kết quả phân tích")
    if not analysis_result or not analysis_result.get("risks"):
        st.info("Chưa có kết quả phân tích. Hãy nhập code, link dự án, hoặc file trước.")
        return

    risks = analysis_result["risks"]
    top = risks[0]
    left, right = st.columns([1, 1])
    with left:
        st.metric("Defect probability", top["probability"])
        st.metric("Risk level", top["severity"])
        st.metric("File focus", top["path"])
        scoring_source = top.get("source_type", "heuristic")
        st.caption("Scoring source: hybrid live triage (heuristic + model-backed probability)" if scoring_source == "model+heuristic" else "Scoring source: heuristic live triage")
        model_probability = top.get("model_probability")
        if model_probability:
            st.caption(f"Model probability: {model_probability}")
        else:
            st.caption("Model probability: not available for this live analysis.")
        source_type = top.get("source_type") or "heuristic"
        st.caption(f"Provenance: {source_type}")
    with right:
        st.markdown("**Analysis source details**")
        st.write(f"Source: {analysis_result.get('source', 'N/A')}")
        st.write(f"Files analyzed: {analysis_result.get('file_count', 0)}")
        if analysis_result.get("excluded_files"):
            st.caption(f"Excluded docs/files: {len(analysis_result['excluded_files'])}")
        for note in analysis_result.get("notes", [])[:6]:
            st.caption(f"- {note}")
        if analysis_result.get("explainability"):
            exp = analysis_result["explainability"] or {}
            st.info(exp.get("message", ""))
            if exp.get("details"):
                st.caption(f"Explainability provenance: {exp['details']}")

    st.markdown("**High-risk files**")
    st.caption("README/docs bị loại khỏi ranking mặc định để ưu tiên source/config có thể chạy được.")
    st.dataframe(pd.DataFrame(risks), use_container_width=True, hide_index=True)
    with st.expander("Why these files are risky", expanded=True):
        for row in risks:
            st.markdown(f"- **{row['path']}** — {row['reason']} ({row['probability']})")
            st.caption(f"Signals: {', '.join(row['signals'])}")
            if row.get("source_type"):
                st.caption(f"Scoring source: {row['source_type']}")
            if row.get("model_probability"):
                st.caption(f"Model probability: {row['model_probability']}")
            for note in row.get("notes", []):
                st.caption(f"  • {note}")


def render_reference_ui() -> None:
    st.subheader("Cách hoạt động")
    steps = [
        "1. Dán code, nhập link repo, hoặc upload file/zip.",
        "2. Live analyzer chấm rủi ro cho source/config và loại README/docs khỏi ranking mặc định.",
        "3. Nếu có model artifact, hệ thống có thể gắn thêm probability từ model.",
        "4. Research view dùng dataset đã xử lý, metrics, commit-text, SHAP và error analysis.",
    ]
    for step in steps:
        st.markdown(f"- {step}")
    st.caption("Advanced chỉ dành cho flow nghiên cứu/debug, không cần cho luồng phân tích thường.")


def _explainability_label(mode: str) -> str:
    return {
        "artifact": "Artifact-backed",
        "partial": "Partial preview",
        "fallback": "Fallback only",
    }.get(mode, mode)


def _resolve_dashboard_view(state: DatasetDashboardState) -> dict[str, Any]:
    selected_model_row = getattr(state, "selected_model_row", {}) or {}
    best_model_row = getattr(state, "best_model_row", {}) or {}
    feature_family = (
        getattr(state, "feature_family", None)
        or selected_model_row.get("feature_family")
        or selected_model_row.get("feature_set")
        or best_model_row.get("feature_family")
        or best_model_row.get("feature_set")
        or "metrics_only"
    )
    paper_metric_columns = list(
        getattr(state, "paper_metric_columns", None)
        or selected_model_row.get("paper_metric_columns")
        or best_model_row.get("paper_metric_columns")
        or []
    )
    commit_text_available = bool(
        getattr(state, "commit_text_available", False)
        or selected_model_row.get("commit_text_available")
        or selected_model_row.get("uses_commit_text")
        or selected_model_row.get("text_feature_column")
        or best_model_row.get("commit_text_available")
        or best_model_row.get("uses_commit_text")
        or best_model_row.get("text_feature_column")
    )
    explanation_mode = getattr(state, "explanation_mode", None) or "artifact"
    notes = list(getattr(state, "notes", []) or [])
    explanation_status = getattr(getattr(state, "explainability", None), "status", None)
    metrics = getattr(state, "metrics", {}) or {}
    return {
        "selected_model_row": selected_model_row,
        "best_model_row": best_model_row,
        "feature_family": str(feature_family),
        "paper_metric_columns": paper_metric_columns,
        "commit_text_available": commit_text_available,
        "explanation_mode": explanation_mode,
        "notes": notes,
        "explanation_status": explanation_status,
        "metrics": metrics,
    }


def render_research_summary(dataset_name: str, selected_model: str | None) -> None:
    if not dataset_name:
        return
    state = get_dashboard_state(dataset_name, selected_model)
    view = _resolve_dashboard_view(state)
    st.subheader("Research / Paper Alignment")

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset", getattr(state, "dataset_name", None) or "N/A")
    c2.metric("Model", getattr(state, "selected_model", None) or "N/A")
    c3.metric("Feature family", view["feature_family"])

    st.caption("Paper-facing workflow: dataset, metrics, commit-text, explainability.")
    if view["metrics"]:
        metric_bits = []
        for key in ["accuracy", "precision", "recall", "f1", "auc"]:
            value = view["metrics"].get(key)
            if value is not None:
                metric_bits.append(f"{key.upper()}={value}")
        if metric_bits:
            st.write(" | ".join(metric_bits))
    if view["paper_metric_columns"]:
        st.write(f"Paper-aligned metrics: {', '.join(view['paper_metric_columns'])}")
    st.write(f"Commit text: {'Yes' if view['commit_text_available'] else 'No'}")
    st.write(f"Explainability: {_explainability_label(view['explanation_mode'])}")
    if view["explanation_status"]:
        st.caption(view["explanation_status"].message)
    for note in view["notes"][:3]:
        st.caption(f"- {note}")

    with st.expander("Quick preview", expanded=False):
        st.write(f"Dataset: {getattr(state, 'dataset_name', None)}")
        st.write(f"Selected model: {getattr(state, 'selected_model', None)}")
        if getattr(state, "impact_rows", None):
            st.markdown("**Commit impact**")
            st.dataframe(pd.DataFrame(state.impact_rows), use_container_width=True, hide_index=True)
        if getattr(state, "global_explainability_rows", None):
            st.markdown("**SHAP summary**")
            st.dataframe(pd.DataFrame(state.global_explainability_rows), use_container_width=True, hide_index=True)


def render_advanced_panel(dataset_options: list[str]) -> tuple[str, str | None]:
    with st.expander("Advanced (optional)", expanded=False):
        st.caption("Dành cho nghiên cứu/debug. Không cần dùng trong luồng phân tích thông thường.")
        if dataset_options:
            dataset_name = st.selectbox("Dataset", dataset_options, index=0, key="advanced_dataset_select")
            selected_model = st.text_input("Model override (optional)", value="", key="advanced_model_override").strip() or None
        else:
            dataset_name = ""
            selected_model = None
            st.info("No baseline datasets found.")
    return dataset_name, selected_model


def render_backend_preview(dataset_name: str, selected_model: str | None) -> None:
    if not dataset_name:
        return
    state = get_dashboard_state(dataset_name, selected_model)
    view = _resolve_dashboard_view(state)
    with st.expander("Backend preview", expanded=False):
        st.write(f"Dataset: {getattr(state, 'dataset_name', None)}")
        st.write(f"Selected model: {getattr(state, 'selected_model', None)}")
        st.write(f"Feature family: {view['feature_family']}")
        st.write(f"Commit text: {'Yes' if view['commit_text_available'] else 'No'}")
        if view["selected_model_row"].get("text_feature_column"):
            st.write(f"Text feature column: {view['selected_model_row']['text_feature_column']}")
        st.write(f"Explainability: {_explainability_label(view['explanation_mode'])}")
        st.dataframe(pd.DataFrame(getattr(state, "ranking_rows", [])), use_container_width=True)
        if getattr(state, "impact_rows", None):
            st.markdown("**Commit impact**")
            st.dataframe(pd.DataFrame(state.impact_rows), use_container_width=True, hide_index=True)
        if getattr(state, "global_explainability_rows", None):
            st.markdown("**SHAP summary**")
            st.dataframe(pd.DataFrame(state.global_explainability_rows), use_container_width=True, hide_index=True)
        if getattr(state, "error_summary_rows", None):
            st.markdown("**Error analysis**")
            st.dataframe(pd.DataFrame(state.error_summary_rows), use_container_width=True, hide_index=True)
        explainability = getattr(state, "explainability", None)
        if explainability and explainability.status:
            st.caption(explainability.status.message)
        if view["notes"]:
            st.caption("Backend notes")
            for note in view["notes"][:3]:
                st.caption(f"- {note}")
        if getattr(state, "error_case_rows", None):
            st.markdown("**Representative error cases**")
            st.dataframe(pd.DataFrame(state.error_case_rows), use_container_width=True, hide_index=True)


def main() -> None:
    dataset_options = get_available_datasets()
    dataset_name, selected_model = render_advanced_panel(dataset_options)

    render_hero()
    source_text, project_link, uploaded_file, analyze = render_input_card()

    if analyze:
        result = build_analysis_result(source_text, project_link, uploaded_file)
        st.session_state["analysis_result"] = {
            "source": result.source,
            "file_count": result.file_count,
            "risks": [row.__dict__ for row in result.risks],
            "notes": result.notes,
            "excluded_files": result.excluded_files,
            "explainability": result.explainability.__dict__ if result.explainability else None,
        }
        _append_recent_analysis(st.session_state["analysis_result"])
        st.session_state["analysis_requested"] = True
    else:
        st.session_state.setdefault("analysis_requested", False)

    analysis_result = st.session_state.get("analysis_result")
    render_status_message()
    render_dashboard_summary(analysis_result)
    render_recent_analyses()
    render_reference_ui()
    render_research_summary(dataset_name, selected_model)
    render_results(analysis_result)
    render_backend_preview(dataset_name, selected_model)


if __name__ == "__main__":
    main()
