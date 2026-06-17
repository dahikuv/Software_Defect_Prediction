"""Theme and style helpers for the Streamlit app."""

from __future__ import annotations

import streamlit as st


LIGHT_COLORS = {
    "bg_primary": "#f7f7f8",
    "bg_secondary": "#eceff3",
    "bg_card": "#ffffff",
    "text_primary": "#111827",
    "text_secondary": "#374151",
    "border": "#d1d5db",
    "accent": "#4f46e5",
    "chart_1": "#4f46e5",
    "chart_2": "#059669",
    "chart_3": "#d97706",
    "chart_4": "#dc2626",
    "chart_5": "#7c3aed",
}

DARK_COLORS = {
    "bg_primary": "#0f1115",
    "bg_secondary": "#1f2430",
    "bg_card": "#171b22",
    "text_primary": "#f9fafb",
    "text_secondary": "#d1d5db",
    "border": "#3f4654",
    "accent": "#6366f1",
    "chart_1": "#6366f1",
    "chart_2": "#10b981",
    "chart_3": "#f59e0b",
    "chart_4": "#ef4444",
    "chart_5": "#8b5cf6",
}


def get_theme_colors() -> dict[str, str]:
    return DARK_COLORS if st.session_state.get("theme", "light") == "dark" else LIGHT_COLORS


def apply_custom_css() -> None:
    colors = get_theme_colors()
    st.markdown(
        f"""
        <style>
        :root {{
            --app-bg: {colors["bg_primary"]};
            --app-bg-secondary: {colors["bg_secondary"]};
            --app-card: {colors["bg_card"]};
            --app-text: {colors["text_primary"]};
            --app-muted: {colors["text_secondary"]};
            --app-border: {colors["border"]};
            --app-accent: {colors["accent"]};
            --app-button-text: #ffffff;
        }}

        .stApp {{
            background: var(--app-bg) !important;
            color: var(--app-text) !important;
        }}

        .main .block-container {{
            max-width: 1280px;
            padding-top: 1.25rem;
            padding-bottom: 3rem;
        }}

        h1, h2, h3, h4, h5, h6,
        p, label, span,
        [data-testid="stMarkdownContainer"],
        [data-testid="stWidgetLabel"],
        [data-testid="stForm"],
        [data-testid="stExpander"] {{
            color: var(--app-text) !important;
        }}

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stCaptionContainer"],
        .st-emotion-cache-1qg05tj {{
            color: var(--app-muted) !important;
            font-size: .95rem !important;
        }}

        .section-divider {{
            height: 1px;
            background: var(--app-border);
            margin: 1.5rem 0;
        }}

        .hero-title {{
            font-size: 2.55rem;
            line-height: 1.1;
            font-weight: 800;
            margin: 0 0 .35rem 0;
            color: var(--app-text) !important;
        }}

        .hero-copy {{
            max-width: 760px;
            color: var(--app-muted) !important;
            font-size: 1.08rem;
            font-weight: 500;
            margin-bottom: .75rem;
        }}

        .section-title {{
            font-size: 1.32rem;
            font-weight: 800;
            color: var(--app-text) !important;
            margin: 0 0 .5rem 0;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--app-card) !important;
            border-color: var(--app-border) !important;
            border-radius: 12px;
        }}

        div[data-testid="stMetric"] {{
            background: var(--app-card) !important;
            border: 1px solid var(--app-border);
            border-radius: 8px;
            padding: .9rem 1rem;
        }}

        div[data-testid="stMetricLabel"] p {{
            color: var(--app-muted) !important;
            font-size: .95rem !important;
            font-weight: 700 !important;
        }}

        div[data-testid="stMetricValue"] {{
            color: var(--app-text) !important;
            font-size: 1.85rem !important;
            font-weight: 800 !important;
        }}

        .stTextArea textarea,
        .stTextInput input,
        .stSelectbox [data-baseweb="select"] {{
            background: var(--app-bg-secondary) !important;
            border-color: var(--app-border) !important;
            color: var(--app-text) !important;
            border-radius: 6px !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
        }}

        .stTextArea textarea::placeholder,
        .stTextInput input::placeholder {{
            color: var(--app-muted) !important;
            opacity: 1 !important;
            font-weight: 500 !important;
        }}

        .stButton button {{
            background: var(--app-accent) !important;
            color: var(--app-button-text) !important;
            border: 0;
            border-radius: 6px;
            font-size: 1.05rem !important;
            font-weight: 800 !important;
            letter-spacing: 0 !important;
            min-height: 2.85rem;
            transition: opacity .2s ease, transform .2s ease;
        }}

        .stButton button p,
        .stButton button span {{
            color: var(--app-button-text) !important;
            font-size: 1.05rem !important;
            font-weight: 800 !important;
        }}

        .stButton button:hover {{
            opacity: .92;
            transform: translateY(-1px);
            color: var(--app-button-text) !important;
            border: 0;
        }}

        [data-testid="stHeader"] {{
            background: var(--app-bg) !important;
        }}

        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {{
            color: var(--app-text) !important;
        }}

        [data-testid="stToolbar"] button,
        [data-testid="stStatusWidget"] button,
        [data-testid="stHeaderActionElements"] button,
        [data-testid="stToolbarActions"] button,
        [data-testid="stBaseButton-header"],
        button[title="Deploy"],
        button[aria-label="Deploy"],
        button[title="Main menu"],
        button[aria-label="Main menu"],
        button[kind="header"] {{
            background: var(--app-card) !important;
            border: 1px solid var(--app-border) !important;
            border-radius: 999px !important;
            color: var(--app-text) !important;
            min-height: 2.35rem !important;
            padding: .45rem .85rem !important;
            font-size: .95rem !important;
            font-weight: 850 !important;
            letter-spacing: 0 !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, .14) !important;
            transition: background .18s ease, border-color .18s ease, transform .18s ease, box-shadow .18s ease !important;
        }}

        [data-testid="stToolbar"] button:hover,
        [data-testid="stStatusWidget"] button:hover,
        [data-testid="stHeaderActionElements"] button:hover,
        [data-testid="stToolbarActions"] button:hover,
        [data-testid="stBaseButton-header"]:hover,
        button[title="Deploy"]:hover,
        button[aria-label="Deploy"]:hover,
        button[title="Main menu"]:hover,
        button[aria-label="Main menu"]:hover,
        button[kind="header"]:hover {{
            background: var(--app-bg-secondary) !important;
            border-color: var(--app-accent) !important;
            color: var(--app-text) !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, .18) !important;
        }}

        [data-testid="stToolbar"] button p,
        [data-testid="stToolbar"] button span,
        [data-testid="stStatusWidget"] button p,
        [data-testid="stStatusWidget"] button span,
        [data-testid="stHeaderActionElements"] button p,
        [data-testid="stHeaderActionElements"] button span,
        [data-testid="stToolbarActions"] button p,
        [data-testid="stToolbarActions"] button span,
        [data-testid="stBaseButton-header"] p,
        [data-testid="stBaseButton-header"] span,
        button[title="Deploy"] p,
        button[title="Deploy"] span,
        button[aria-label="Deploy"] p,
        button[aria-label="Deploy"] span,
        button[kind="header"] p,
        button[kind="header"] span {{
            color: var(--app-text) !important;
            font-size: .95rem !important;
            font-weight: 850 !important;
        }}

        [data-testid="stToolbar"] svg,
        [data-testid="stHeader"] svg,
        [data-testid="stStatusWidget"] svg,
        [data-testid="stHeaderActionElements"] svg,
        [data-testid="stToolbarActions"] svg,
        [data-testid="stBaseButton-header"] svg,
        button[title="Deploy"] svg,
        button[aria-label="Deploy"] svg,
        button[title="Main menu"] svg,
        button[aria-label="Main menu"] svg,
        button[kind="header"] svg {{
            color: var(--app-text) !important;
            fill: var(--app-text) !important;
            stroke: var(--app-text) !important;
            stroke-width: 2.4px !important;
        }}

        button[title="Main menu"],
        button[aria-label="Main menu"] {{
            background: var(--app-card) !important;
            border: 1px solid var(--app-border) !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, .14) !important;
            border-radius: 999px !important;
            min-width: 2.85rem !important;
            min-height: 2.35rem !important;
            padding: .35rem .75rem !important;
            margin-left: .35rem !important;
        }}

        button[title="Main menu"] *,
        button[aria-label="Main menu"] * {{
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            outline: 0 !important;
        }}

        button[title="Main menu"]:hover,
        button[aria-label="Main menu"]:hover {{
            background: var(--app-bg-secondary) !important;
            border: 1px solid var(--app-accent) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, .16) !important;
            transform: translateY(-1px) !important;
        }}

        button[title="Main menu"] svg,
        button[aria-label="Main menu"] svg {{
            width: 14px !important;
            height: 14px !important;
            color: var(--app-muted) !important;
            fill: var(--app-muted) !important;
            stroke: var(--app-muted) !important;
            stroke-width: 1.6px !important;
        }}

        [data-testid="stFileUploader"] {{
            color: var(--app-text) !important;
        }}

        [data-testid="stFileUploader"] section {{
            background: var(--app-bg-secondary) !important;
            border: 1px dashed var(--app-border) !important;
            border-radius: 8px !important;
        }}

        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] small {{
            color: var(--app-text) !important;
            font-size: .98rem !important;
            font-weight: 650 !important;
        }}

        [data-testid="stFileUploader"] button,
        [data-testid="stFormSubmitButton"] button {{
            background: var(--app-accent) !important;
            color: var(--app-button-text) !important;
            border: 0 !important;
            border-radius: 6px !important;
            font-size: 1.08rem !important;
            font-weight: 850 !important;
            min-height: 2.9rem !important;
        }}

        [data-testid="stFileUploader"] button p,
        [data-testid="stFileUploader"] button span,
        [data-testid="stFormSubmitButton"] button p,
        [data-testid="stFormSubmitButton"] button span {{
            color: var(--app-button-text) !important;
            font-size: 1.08rem !important;
            font-weight: 850 !important;
        }}

        [data-testid="stFileUploader"] svg,
        [data-testid="stFormSubmitButton"] svg,
        .stButton button svg {{
            color: var(--app-button-text) !important;
            fill: var(--app-button-text) !important;
            stroke: var(--app-button-text) !important;
        }}

        [data-testid="stToggle"] {{
            background: var(--app-card) !important;
            border: 1px solid var(--app-border) !important;
            border-radius: 999px !important;
            padding: .35rem .65rem !important;
        }}

        [data-testid="stToggle"] label,
        [data-testid="stToggle"] p,
        [data-testid="stToggle"] span {{
            color: var(--app-text) !important;
            font-weight: 800 !important;
            font-size: .98rem !important;
        }}

        [data-testid="stToggle"] div[role="switch"] {{
            border: 1px solid var(--app-border) !important;
        }}

        .stDataFrame {{
            border: 1px solid var(--app-border);
            border-radius: 8px;
            background: var(--app-card) !important;
            color: var(--app-text) !important;
        }}

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {{
            color: var(--app-text) !important;
        }}

        .streamlit-expanderHeader {{
            background: var(--app-card) !important;
            border: 1px solid var(--app-border) !important;
            border-radius: 6px !important;
            color: var(--app-text) !important;
            font-size: 1rem !important;
            font-weight: 800 !important;
        }}

        .streamlit-expanderHeader p,
        .streamlit-expanderHeader span {{
            color: var(--app-text) !important;
            font-weight: 800 !important;
        }}

        div[data-testid="stAlert"] {{
            background: var(--app-card) !important;
            border: 1px solid var(--app-border);
            border-left: 4px solid var(--app-accent);
            color: var(--app-text) !important;
        }}

        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] span {{
            color: var(--app-text) !important;
            font-weight: 600 !important;
        }}

        @media (max-width: 720px) {{
            .main .block-container {{
                padding-left: .85rem;
                padding-right: .85rem;
            }}
            .hero-title {{
                font-size: 1.75rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_divider() -> None:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def render_section_title(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
