"""Streamlit view layer for the defect prediction demo."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:  # pragma: no cover - optional UI dependency fallback
    go = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.controllers import DatasetDashboardState, build_dashboard_state, list_available_datasets
from src.app.services.dataset_service import COMMIT_LEVEL_DATASETS, DATASET_GRANULARITY, MODULE_LEVEL_DATASETS
from src.app.services.repo_analysis_service import build_analysis_result
from src.app.theme import apply_custom_css, get_theme_colors, render_section_divider, render_section_title
from src.utils.coercion import coerce_bool
from src.utils.io import read_csv
from src.utils.logging import get_logger
from src.utils.paths import RESULTS_TABLES_DIR

st.set_page_config(page_title="Defect Risk Analyzer", layout="wide", initial_sidebar_state="collapsed")

logger = get_logger(__name__)


@st.cache_data(show_spinner=False)
def get_available_datasets() -> list[str]:
    return list_available_datasets()


@st.cache_data(show_spinner=False)
def get_dashboard_state(dataset_name: str, selected_model: str | None) -> DatasetDashboardState:
    return build_dashboard_state(dataset_name, selected_model)


def _safe_probability(value: Any) -> int:
    try:
        return max(0, min(100, int(float(str(value).strip().rstrip("%")))))
    except (TypeError, ValueError):
        return 0


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except TypeError:
        pass
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


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


def create_risk_trend_chart(risks: list[dict[str, Any]]):
    values = [_safe_probability(row.get("probability")) for row in risks[:10]] or [0]
    colors = get_theme_colors()
    if go is None:
        return pd.DataFrame({"risk": values})

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(values) + 1)),
            y=values,
            mode="lines+markers",
            line={"color": colors["chart_1"], "width": 3},
            marker={"size": 8, "color": colors["chart_1"]},
            fill="tozeroy",
            fillcolor="rgba(79, 70, 229, 0.10)",
        )
    )
    fig.update_layout(
        height=300,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": colors["text_secondary"]},
        xaxis={"title": "File index", "showgrid": True, "gridcolor": colors["border"]},
        yaxis={"title": "Risk %", "range": [0, 100], "showgrid": True, "gridcolor": colors["border"]},
        showlegend=False,
    )
    return fig


def create_distribution_chart(risks: list[dict[str, Any]]):
    severities = ["Critical", "High", "Medium", "Low"]
    counts = pd.Series([row.get("severity", "Low") for row in risks]).value_counts()
    values = [int(counts.get(severity, 0)) for severity in severities]
    colors = get_theme_colors()
    if go is None:
        return pd.DataFrame({"count": values}, index=severities)

    fig = go.Figure(
        data=[
            go.Bar(
                x=severities,
                y=values,
                marker={
                    "color": [colors["chart_4"], colors["chart_3"], colors["chart_2"], colors["chart_1"]],
                    "line": {"width": 0},
                },
            )
        ]
    )
    fig.update_layout(
        height=300,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": colors["text_secondary"]},
        xaxis={"showgrid": False},
        yaxis={"title": "Files", "showgrid": True, "gridcolor": colors["border"]},
        showlegend=False,
    )
    return fig


def render_chart(chart) -> None:
    if go is not None and hasattr(chart, "to_plotly_json"):
        st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
    elif isinstance(chart, pd.DataFrame):
        if "risk" in chart.columns:
            st.line_chart(chart)
        else:
            st.bar_chart(chart)


def render_top_bar() -> None:
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"

    spacer, control = st.columns([6, 1.4])
    with spacer:
        st.caption("Software Defect Prediction")
    with control:
        current_is_dark = st.session_state["theme"] == "dark"
        selected_is_dark = st.toggle("Dark mode", value=current_is_dark, key="theme_toggle")
        selected_theme = "dark" if selected_is_dark else "light"
        if selected_theme != st.session_state["theme"]:
            st.session_state["theme"] = selected_theme
            st.rerun()


def render_hero() -> None:
    st.markdown('<div class="hero-title">Defect Risk Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-copy">Analyze source code, repositories, and saved research artifacts with a compact risk dashboard.</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Goal", "Risk ranking")
    c2.metric("Inputs", "Code / Repo / File")
    c3.metric("Output", "Actionable triage")


def render_input_card() -> tuple[str, str, Any, bool]:
    with st.container(border=True):
        render_section_title("Analysis input")
        with st.form("analysis_form", clear_on_submit=False):
            left, right = st.columns([2, 1])
            with left:
                source_text = st.text_area(
                    "Paste code",
                    placeholder="Paste source code or a compact project snippet...",
                    height=220,
                    key="source_text_input",
                )
                project_link = st.text_input(
                    "Repository link",
                    placeholder="https://github.com/owner/project",
                    key="project_link_input",
                )
            with right:
                uploaded = st.file_uploader("Upload file or zip", type=["py", "txt", "md", "json", "csv", "zip"])
                st.caption("Use one or more inputs, then run the analyzer.")
                analyze = st.form_submit_button("Analyze project", width="stretch")
    return source_text, project_link, uploaded, analyze


def render_status_message() -> None:
    if st.session_state.get("analysis_requested"):
        st.success("Analysis complete.")


def render_dashboard_summary(analysis_result: dict[str, Any] | None) -> None:
    risks = analysis_result.get("risks", []) if analysis_result else []
    file_count = analysis_result.get("file_count", 0) if analysis_result else 0
    critical_count = sum(1 for row in risks if row.get("severity") == "Critical")
    top_risk = risks[0].get("probability", "0%") if risks else "0%"

    files_with_commits = sum(
        1 for row in risks
        if any(str(s).startswith("commit_msg:") for s in row.get("signals", []))
    )
    is_hybrid = any(
        "hybrid" in str(row.get("source_type", "")).lower() for row in risks
    )

    with st.container(border=True):
        render_section_title("Dashboard summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Files analyzed", str(file_count))
        c2.metric("Ranked files", str(len(risks)))
        c3.metric("Critical files", str(critical_count))
        c4.metric("Top risk", top_risk)
        c5.metric(
            "Commit messages",
            f"{files_with_commits}/{len(risks)}" if risks else "0/0",
            help="Files with commit messages used by the hybrid model",
        )

        if is_hybrid:
            st.success(
                f"**Hybrid Model Active** — Commit message TF-IDF features were used for {files_with_commits} files. "
                "Commit messages are extracted from git history and combined with software metrics for improved defect prediction.",
                icon=":material/auto_awesome:",
            )

        left, right = st.columns(2)
        with left:
            st.markdown("**Risk trends**")
            render_chart(create_risk_trend_chart(risks))
        with right:
            st.markdown("**Defect distribution**")
            render_chart(create_distribution_chart(risks))


def render_recent_analyses() -> None:
    with st.container(border=True):
        render_section_title("Recent analyses")
        history = st.session_state.get("recent_analyses", [])
        if not history:
            st.info("No analyses in this session yet.")
            return
        st.dataframe(pd.DataFrame(history), width="stretch", hide_index=True)


def render_reference_ui() -> None:
    with st.container(border=True):
        render_section_title("How it works")
        steps = pd.DataFrame(
            [
                {"Step": "1", "Stage": "Input", "Output": "Source text, repository files, or uploaded archive"},
                {"Step": "2", "Stage": "Scoring", "Output": "Heuristic and model-backed risk signals"},
                {"Step": "3", "Stage": "Ranking", "Output": "High-risk files ordered by defect probability"},
                {"Step": "4", "Stage": "Review", "Output": "Signals, provenance, SHAP, and artifact previews"},
            ]
        )
        st.dataframe(steps, width="stretch", hide_index=True)


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
    commit_text_available = any(
        [
            coerce_bool(getattr(state, "commit_text_available", False)),
            coerce_bool(selected_model_row.get("commit_text_available")),
            coerce_bool(selected_model_row.get("uses_commit_text")),
            coerce_bool(best_model_row.get("commit_text_available")),
            coerce_bool(best_model_row.get("uses_commit_text")),
        ]
    )
    return {
        "selected_model_row": selected_model_row,
        "best_model_row": best_model_row,
        "feature_family": str(feature_family),
        "paper_metric_columns": paper_metric_columns,
        "commit_text_available": commit_text_available,
        "explanation_mode": getattr(state, "explanation_mode", None) or "artifact",
        "notes": list(getattr(state, "notes", []) or []),
        "explanation_status": getattr(getattr(state, "explainability", None), "status", None),
        "metrics": getattr(state, "metrics", {}) or {},
    }




IMPACT_TABLE_PATH = RESULTS_TABLES_DIR / "commit_message_impact.csv"
IMPACT_BRANCH_SUMMARY_PATH = RESULTS_TABLES_DIR / "commit_message_impact_branch_summary.csv"
IMPACT_DATASET_SUMMARY_PATH = RESULTS_TABLES_DIR / "commit_message_impact_summary.csv"


@st.cache_data(show_spinner=False)
def load_commit_impact_tables() -> dict[str, pd.DataFrame]:
    return {
        "impact": read_csv(IMPACT_TABLE_PATH) if IMPACT_TABLE_PATH.exists() else pd.DataFrame(),
        "branch_summary": read_csv(IMPACT_BRANCH_SUMMARY_PATH) if IMPACT_BRANCH_SUMMARY_PATH.exists() else pd.DataFrame(),
        "dataset_summary": read_csv(IMPACT_DATASET_SUMMARY_PATH) if IMPACT_DATASET_SUMMARY_PATH.exists() else pd.DataFrame(),
    }


def _annotate_granularity(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "dataset_name" not in df.columns:
        return df
    df = df.copy()
    df["granularity"] = df["dataset_name"].map(DATASET_GRANULARITY).fillna("module")
    return df


def render_commit_impact_section() -> None:
    tables = load_commit_impact_tables()
    impact_df = _annotate_granularity(tables["impact"])
    dataset_summary_df = _annotate_granularity(tables["dataset_summary"])
    branch_summary_df = tables["branch_summary"].copy()

    with st.container(border=True):
        render_section_title("Commit message impact")
        if impact_df.empty:
            st.info("Run scripts/run_commit_message_impact.py to populate this view.")
            return

        st.warning("Commit-message impact is computed only for commit-level datasets that contain commit text (openstack, qt, jitfine). PROMISE module-level datasets are excluded by design because they do not have commit messages.", icon=":material/info:")
        st.caption("Hybrid metrics + commit text vs metrics-only baseline. Datasets are split by granularity (module-level NASA/PROMISE vs commit-level JITLine).")

        if not branch_summary_df.empty:
            st.markdown("**Branch averages**")
            st.dataframe(branch_summary_df, width="stretch", hide_index=True)

        granularity_groups = sorted(impact_df["granularity"].unique().tolist())
        for granularity in granularity_groups:
            st.markdown(f"**{granularity.title()}-level datasets**")
            granularity_impact = impact_df[impact_df["granularity"] == granularity]
            if not dataset_summary_df.empty and "granularity" in dataset_summary_df.columns:
                granularity_summary = dataset_summary_df[dataset_summary_df["granularity"] == granularity]
                if not granularity_summary.empty:
                    st.dataframe(granularity_summary, width="stretch", hide_index=True)
            st.dataframe(granularity_impact, width="stretch", hide_index=True)


def render_research_summary(dataset_name: str, selected_model: str | None) -> None:
    if not dataset_name:
        return
    state = get_dashboard_state(dataset_name, selected_model)
    view = _resolve_dashboard_view(state)

    with st.container(border=True):
        render_section_title("Research summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dataset", getattr(state, "dataset_name", None) or "N/A")
        c2.metric("Model", getattr(state, "selected_model", None) or "N/A")
        c3.metric("Feature family", view["feature_family"])
        c4.metric("Explainability", _explainability_label(view["explanation_mode"]))

        metric_cols = st.columns(5)
        for idx, key in enumerate(["accuracy", "precision", "recall", "f1", "auc"]):
            metric_cols[idx].metric(key.upper(), _format_metric(view["metrics"].get(key)))

        detail_left, detail_right = st.columns([1, 1])
        with detail_left:
            if view["paper_metric_columns"]:
                st.write("Metric columns: " + ", ".join(view["paper_metric_columns"]))
            st.write(f"Commit text: {'Available' if view['commit_text_available'] else 'Not available'}")
            if view["selected_model_row"].get("selection_data_source"):
                st.write(f"Selection source: {view['selected_model_row']['selection_data_source']}")
            if coerce_bool(view["selected_model_row"].get("test_metrics_report_only", False)):
                st.caption("Displayed test metrics are report-only, not selection criteria.")
            if view["explanation_status"]:
                st.caption(view["explanation_status"].message)
        with detail_right:
            for note in view["notes"][:4]:
                st.caption(note)

        with st.expander("Artifact preview", expanded=False):
            if getattr(state, "impact_rows", None):
                st.markdown("**Commit impact**")
                st.dataframe(pd.DataFrame(state.impact_rows), width="stretch", hide_index=True)
            if getattr(state, "global_explainability_rows", None):
                st.markdown("**SHAP summary**")
                st.dataframe(pd.DataFrame(state.global_explainability_rows), width="stretch", hide_index=True)


def _extract_commit_messages_from_risks(risks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract commit message info from risk rows for dedicated display."""
    commit_rows = []
    for row in risks:
        signals = row.get("signals", [])
        commit_msg = ""
        for signal in signals:
            if signal.startswith("commit_msg:"):
                commit_msg = signal.replace("commit_msg:", "").strip().strip('"')
                break
        if commit_msg:
            commit_rows.append({
                "File": row.get("path", "N/A"),
                "Commit Messages": commit_msg,
                "Defect Probability": row.get("probability", "0%"),
                "Severity": row.get("severity", "N/A"),
            })
    return commit_rows


def render_results(analysis_result: dict[str, Any] | None) -> None:
    with st.container(border=True):
        render_section_title("Analysis results")
        if not analysis_result or not analysis_result.get("risks"):
            st.info("Run an analysis to populate the risk table.")
            return

        risks = analysis_result["risks"]
        top = risks[0]

        is_hybrid = any(
            "hybrid" in str(row.get("source_type", "")).lower()
            for row in risks
        )

        if is_hybrid:
            st.success(
                "Hybrid model active: predictions use **software metrics + commit message TF-IDF features**. "
                "Commit messages are extracted from git history and transformed into 1000 TF-IDF features that improve defect detection.",
                icon=":material/check_circle:",
            )
        else:
            st.warning(
                "This live repo/upload analysis is a heuristic risk ranking, not a defect prediction model. "
                "It scores files by proxy signals (size, complexity-like patterns) and is intended for triage only. "
                "Use the dataset and model views below for the research-grade SDP results.",
                icon=":material/warning:",
            )

        left, right = st.columns([1, 1])
        with left:
            c1, c2 = st.columns(2)
            c1.metric("Defect probability", top.get("probability", "0%"))
            c2.metric("Risk level", top.get("severity", "N/A"))
            st.write(f"Top file: `{top.get('path', 'N/A')}`")
            scoring_label = "Hybrid (Metrics + Commit Text)" if is_hybrid else top.get("source_type", "heuristic")
            st.caption(f"Scoring source: {scoring_label}")
            if top.get("model_probability"):
                st.caption(f"Model probability: {top['model_probability']}")
        with right:
            st.write(f"Source: {analysis_result.get('source', 'N/A')}")
            st.write(f"Files analyzed: {analysis_result.get('file_count', 0)}")
            if analysis_result.get("excluded_files"):
                st.caption(f"Excluded files: {len(analysis_result['excluded_files'])}")
            if analysis_result.get("explainability"):
                exp = analysis_result["explainability"] or {}
                st.info(exp.get("message", ""))

        commit_rows = _extract_commit_messages_from_risks(risks)
        if commit_rows:
            st.markdown("---")
            st.markdown("### Commit Message Analysis")
            st.caption(
                f"Commit messages extracted from git history for **{len(commit_rows)}/{len(risks)}** files. "
                "These messages are converted to TF-IDF features and used by the hybrid model to improve defect prediction accuracy."
            )
            commit_df = pd.DataFrame(commit_rows)
            st.dataframe(commit_df, width="stretch", hide_index=True)

            with st.expander("How commit messages improve prediction", expanded=False):
                st.markdown(
                    "The hybrid model combines **22 software metrics** (LOC, cyclomatic complexity, etc.) "
                    "with **1000 TF-IDF features** extracted from commit messages.\n\n"
                    "Commit messages contain valuable signals:\n"
                    "- Words like *fix*, *bug*, *error*, *crash* indicate defect-prone areas\n"
                    "- Frequent *refactor* or *cleanup* commits suggest complex, evolving code\n"
                    "- Short/vague messages may correlate with hasty changes\n\n"
                    "Research shows this hybrid approach improves F1-score by 5-15% over metrics-only models "
                    "on commit-level datasets (openstack, qt, jitfine)."
                )

        st.markdown("---")
        st.markdown("**High-risk files**")
        risk_table = pd.DataFrame(risks)
        display_columns = ["path", "probability", "severity", "source_type"]
        if not risk_table.empty:
            risk_table["model_type"] = risk_table.get("source_type", "heuristic").map(
                lambda value: "Hybrid (Metrics+Commit)" if "hybrid" in str(value or "").lower()
                else ("Model" if "model" in str(value or "").lower() else "Heuristic")
            )
        st.dataframe(risk_table, width="stretch", hide_index=True)

        with st.expander("Risk reasoning & commit signals", expanded=True):
            for row in risks:
                st.markdown(f"**{row.get('path', 'N/A')}** - {row.get('probability', '0%')}")
                st.caption(row.get("reason", "No reason available."))
                signals = row.get("signals", [])
                commit_signals = [s for s in signals if s.startswith("commit_msg:")]
                other_signals = [s for s in signals if not s.startswith("commit_msg:")]
                if commit_signals:
                    for cs in commit_signals:
                        msg = cs.replace("commit_msg:", "").strip().strip('"')
                        st.markdown(f"  *Commit messages:* `{msg}`")
                if other_signals:
                    st.caption("Signals: " + ", ".join(other_signals))
                for note in row.get("notes", []):
                    st.caption(note)


def render_advanced_panel(dataset_options: list[str]) -> tuple[str, str | None]:
    with st.expander("Advanced settings", expanded=False):
        if dataset_options:
            dataset_name = st.selectbox("Dataset", dataset_options, index=0, key="advanced_dataset_select")
            model_options = []
            try:
                state = get_dashboard_state(dataset_name, None)
                model_options = list(getattr(state, "model_options", []) or [])
            except Exception as exc:
                logger.exception("Failed to load model options for dataset=%s", dataset_name)
                st.caption(f"Could not load model options for {dataset_name}: {exc}")
                model_options = []

            if model_options:
                selected_model = st.selectbox("Model", ["Auto", *model_options], index=0, key="advanced_model_select")
                selected_model = None if selected_model == "Auto" else selected_model
            else:
                selected_model = st.text_input("Model override", value="", key="advanced_model_override").strip() or None
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
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Dataset", getattr(state, "dataset_name", None) or "N/A")
        c2.metric("Selected model", getattr(state, "selected_model", None) or "N/A")
        c3.metric("Feature family", view["feature_family"])
        c4.metric("Commit text", "Yes" if view["commit_text_available"] else "No")

        if view["selected_model_row"].get("text_feature_column"):
            st.write(f"Text feature column: `{view['selected_model_row']['text_feature_column']}`")
        st.write(f"Explainability: {_explainability_label(view['explanation_mode'])}")
        st.dataframe(pd.DataFrame(getattr(state, "ranking_rows", [])), width="stretch")

        if getattr(state, "error_summary_rows", None):
            st.markdown("**Error analysis**")
            st.dataframe(pd.DataFrame(state.error_summary_rows), width="stretch", hide_index=True)
        if getattr(state, "error_case_rows", None):
            st.markdown("**Representative error cases**")
            st.dataframe(pd.DataFrame(state.error_case_rows), width="stretch", hide_index=True)


def main() -> None:
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"
    apply_custom_css()

    dataset_options = get_available_datasets()
    dataset_name, selected_model = render_advanced_panel(dataset_options)

    render_top_bar()
    render_hero()
    render_section_divider()
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
    render_section_divider()
    render_dashboard_summary(analysis_result)
    render_section_divider()
    render_recent_analyses()
    render_section_divider()
    render_reference_ui()
    render_section_divider()
    render_research_summary(dataset_name, selected_model)
    render_section_divider()
    render_commit_impact_section()
    render_section_divider()
    render_results(analysis_result)
    render_backend_preview(dataset_name, selected_model)


if __name__ == "__main__":
    main()
