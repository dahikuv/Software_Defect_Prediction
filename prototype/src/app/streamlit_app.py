"""Streamlit view layer - Defect Risk Analyzer.

Refined research-paper demo UI. Single-purpose: paste a GitHub URL,
get risk analysis. Built around calm visual hierarchy and clear data presentation.
"""
from __future__ import annotations

import inspect
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None

try:
    import streamlit.components.v1 as components
except Exception:
    components = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.services.repo_analysis_service import build_analysis_result
from src.utils.logging import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title="Defect Risk Analyzer",
    page_icon="\U0001f52c",
    layout="wide",
    initial_sidebar_state="collapsed",
)

for _k, _v in {"theme": "light", "analysis_result": None, "analysis_error": None}.items():
    st.session_state.setdefault(_k, _v)

LIGHT = {
    "bg": "#FFFFFF",
    "surface": "#FFFFFF",
    "surface_2": "#F6F8FA",
    "surface_3": "#EAEEF2",
    "border": "#D0D7DE",
    "border_strong": "#8C959F",
    "text_primary": "#1F2328",
    "text_secondary": "#57606A",
    "text_muted": "#8C959F",
    "primary": "#0969DA",
    "primary_hover": "#0860C7",
    "primary_active": "#0757B3",
    "primary_fg": "#FFFFFF",
    "primary_subtle": "rgba(9, 105, 218, 0.08)",
    "primary_border": "rgba(9, 105, 218, 0.4)",
    "success": "#1A7F37",
    "warning": "#9A6700",
    "danger": "#CF222E",
    "critical": "#CF222E",
    "high": "#BC4C00",
    "medium": "#9A6700",
    "low": "#1A7F37",
    "critical_bg": "rgba(207, 34, 46, 0.10)",
    "high_bg": "rgba(188, 76, 0, 0.10)",
    "medium_bg": "rgba(154, 103, 0, 0.10)",
    "low_bg": "rgba(26, 127, 55, 0.10)",
    "shadow_xs": "0 1px 0 rgba(31,35,40,0.04)",
    "shadow_sm": "0 1px 2px rgba(31,35,40,0.06)",
    "shadow_md": "0 3px 6px rgba(31,35,40,0.10)",
    "shadow_lg": "0 8px 24px rgba(31,35,40,0.12)",
    "code_bg": "#F6F8FA",
    "scrollbar_thumb": "rgba(31,35,40,0.20)",
    "scrollbar_thumb_hover": "rgba(31,35,40,0.35)",
    "topbar_border": "rgba(31,35,40,0.08)",
    "hairline": "rgba(31,35,40,0.06)",
    "hero_grad": "linear-gradient(135deg, #E6F0FF 0%, #F6F8FA 100%)",
    "empty_state_grad": "linear-gradient(90deg, #0969DA 0%, #0860C7 100%)"
}

DARK = {
    "bg": "#0B0F14",
    "surface": "#151B23",
    "surface_2": "#1E2530",
    "surface_3": "#2A323D",
    "border": "#3A4452",
    "border_strong": "#7C8898",
    "text_primary": "#F0F4F8",
    "text_secondary": "#A0ADC0",
    "text_muted": "#7C8898",
    "primary": "#4F95FF",
    "primary_hover": "#6BA5FF",
    "primary_active": "#3B82F6",
    "primary_fg": "#FFFFFF",
    "primary_subtle": "rgba(79, 149, 255, 0.18)",
    "primary_border": "rgba(79, 149, 255, 0.45)",
    "success": "#4FCB6A",
    "warning": "#E0B33A",
    "danger": "#FF6B6B",
    "critical": "#FF6B6B",
    "high": "#FF9933",
    "medium": "#E0B33A",
    "low": "#4FCB6A",
    "critical_bg": "rgba(255, 107, 107, 0.18)",
    "high_bg": "rgba(255, 153, 51, 0.18)",
    "medium_bg": "rgba(224, 179, 58, 0.18)",
    "low_bg": "rgba(79, 203, 106, 0.18)",
    "shadow_xs": "0 1px 0 rgba(0,0,0,0.20)",
    "shadow_sm": "0 1px 2px rgba(0,0,0,0.30), 0 0 0 1px rgba(79, 149, 255, 0.04)",
    "shadow_md": "0 4px 12px rgba(0,0,0,0.45), 0 0 0 1px rgba(79, 149, 255, 0.06)",
    "shadow_lg": "0 8px 24px rgba(0,0,0,0.55), 0 0 0 1px rgba(79, 149, 255, 0.08)",
    "code_bg": "#1E2530",
    "scrollbar_thumb": "rgba(160, 173, 192, 0.20)",
    "scrollbar_thumb_hover": "rgba(160, 173, 192, 0.35)",
    "topbar_border": "rgba(160, 173, 192, 0.18)",
    "hairline": "rgba(160, 173, 192, 0.12)",
    "hero_grad": "linear-gradient(135deg, rgba(79, 149, 255, 0.12) 0%, rgba(40, 55, 80, 0.4) 100%)",
    "empty_state_grad": "linear-gradient(90deg, #4F95FF 0%, #6BA5FF 100%)",
}




def _t():
    return DARK if st.session_state.get("theme", "light") == "dark" else LIGHT


def _theme_js():
    """Inject JS that:
       - reads theme from localStorage on load (or falls back to current attribute)
       - listens for clicks on [data-theme-toggle] and flips data-theme on <html>
       - persists choice to localStorage
       - updates Plotly chart text/grid colors to match the new theme
    No Streamlit rerun, no page reload - truly instant."""
    if components is None:
        return
    components.html(
        """<script>
        (function() {
            var KEY = "sdp-theme";
            var pdoc = window.parent.document;
            var pwin = window.parent;
            var root = pdoc.documentElement;
            try {
                var saved = pwin.localStorage.getItem(KEY);
                if (saved) root.setAttribute("data-theme", saved);
            } catch (e) {}
            // Light/dark palettes for Plotly chart text/grid
            var PALETTE = {
                light: { text: "#1F2328", text2: "#57606A", grid: "rgba(128,128,128,0.12)" },
                dark:  { text: "#F0F4F8", text2: "#A0ADC0", grid: "rgba(160,173,192,0.18)" }
            };
            function updateCharts(theme) {
                var p = PALETTE[theme] || PALETTE.light;
                var charts = pdoc.querySelectorAll(".js-plotly-plot");
                for (var i = 0; i < charts.length; i++) {
                    var gd = charts[i];
                    var Plotly = pwin.Plotly || (pwin.parent && pwin.parent.Plotly);
                    if (!Plotly || !Plotly.relayout) continue;
                    try {
                        Plotly.relayout(gd, {
                            "font.color": p.text,
                            "title.font.color": p.text,
                            "xaxis.tickfont.color": p.text2,
                            "yaxis.tickfont.color": p.text2,
                            "xaxis.title.font.color": p.text,
                            "yaxis.title.font.color": p.text,
                            "xaxis.gridcolor": p.grid,
                            "yaxis.gridcolor": p.grid,
                            "legend.font.color": p.text
                        });
                        // Update donut/pie label color too (uses insidetextfont + outsidetextfont)
                        try {
                            Plotly.relayout(gd, {
                                "insidetextfont.color": "#FFFFFF",
                                "outsidetextfont.color": p.text2
                            });
                        } catch (e2) {}
                    } catch (e) {}
                }
            }
            function applyTheme(theme) {
                root.setAttribute("data-theme", theme);
                try { pwin.localStorage.setItem(KEY, theme); } catch (e) {}
                // Re-apply charts after DOM has settled on the new CSS vars.
                pwin.setTimeout(function() { updateCharts(theme); }, 0);
            }
            // Listen on parent document (the actual page, not this iframe)
            pdoc.addEventListener("click", function(e) {
                var btn = e.target && e.target.closest
                    ? e.target.closest("[data-theme-toggle]")
                    : null;
                if (!btn) return;
                e.preventDefault();
                var cur = root.getAttribute("data-theme") || "light";
                var next = cur === "light" ? "dark" : "light";
                applyTheme(next);
            }, true);
            // Also re-style charts once on initial load (matches saved theme)
            pwin.setTimeout(function() {
                var cur = root.getAttribute("data-theme") || "light";
                updateCharts(cur);
            }, 250);
        })();
        </script>""",
        height=0,
    )

def _inject_css():
    light_vars = "; ".join(f"--{k}: {v}" for k, v in LIGHT.items())
    dark_vars  = "; ".join(f"--{k}: {v}" for k, v in DARK.items())
    css = """<style>
:root, :root[data-theme="light"] { """ + light_vars + """ }
:root[data-theme="dark"] { """ + dark_vars + """ }
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text_primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", system-ui, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    transition: background-color 0.2s ease, color 0.2s ease;
}
.main .block-container {
    max-width: 1120px;
    padding-top: 0;
    padding-bottom: 5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
h1, h2, h3, h4, p, label, span, div { color: var(--text_primary); }
[data-testid="stCaptionContainer"] p { color: var(--text_secondary) !important; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.stDeployButton { display: none; }

/* TOPBAR - centered brand, larger, balanced; toggle floats top-right */
.topbar {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1.75rem 0 1.5rem 0;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--topbar_border);
    position: relative;
}
.topbar-brand {
    display: inline-flex; align-items: center; gap: 0.85rem;
    font-weight: 700; font-size: 1.4rem;
    color: var(--text_primary) !important;
    letter-spacing: -0.02em;
    line-height: 1;
}
.topbar-brand-mark {
    width: 36px; height: 36px;
    border-radius: 9px;
    background: var(--primary);
    display: inline-flex; align-items: center; justify-content: center;
    color: var(--primary_fg);
    box-shadow: var(--shadow_sm);
}
.topbar-brand-mark svg { display: block; }
.topbar-brand-name { font-weight: 700; }
.topbar-brand-tag {
    font-size: 0.65rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text_muted) !important;
    padding: 3px 8px;
    border-radius: 999px;
    background: var(--surface_2);
    border: 1px solid var(--border);
    margin-left: 0.25rem;
}
.topbar-theme-btn {
    display: inline-flex; align-items: center; justify-content: center;
    width: 34px; height: 34px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text_secondary);
    text-decoration: none;
    transition: all 0.15s ease;
}
.topbar-theme-btn:hover {
    background: var(--surface_2);
    border-color: var(--border_strong);
    color: var(--text_primary);
}
.topbar-theme-btn svg { display: block; }
.topbar-theme-btn:focus-visible { outline: 2px solid var(--primary); outline-offset: 2px; }

/* HERO */
.hero {
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 2.5rem;
    max-width: 920px;
    background: var(--hero_grad);
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: var(--shadow_sm);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -50%;
    right: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(circle, var(--primary) 0%, transparent 70%);
    pointer-events: none;
    opacity: 0.12;
}
.hero-eyebrow {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    background: var(--primary_subtle);
    color: var(--primary) !important;
    border-radius: 4px;
    font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1.1;
    letter-spacing: -0.025em;
    color: var(--text_primary) !important;
    margin: 0 0 0.75rem 0;
}
.hero p {
    font-size: 1.05rem;
    line-height: 1.55;
    color: var(--text_secondary) !important;
    margin: 0;
    max-width: 600px;
}

/* INPUT ROW */
.input-row {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
    max-width: 720px;
}
.stTextInput input {
    background: var(--surface) !important;
    color: var(--text_primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.65rem 0.9rem !important;
    font-size: 0.95rem !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: var(--shadow_xs) !important;
    caret-color: var(--primary) !important;
}
.stTextInput input::placeholder { color: var(--text_muted) !important; }
.stTextInput input:hover { border-color: var(--border_strong) !important; }
.stTextInput input:focus {
    border-color: var(--primary) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px var(--primary_subtle) !important;
}
.stTextInput label { display: none; }
.input-row .stButton button {
    background: var(--primary) !important;
    color: var(--primary_fg) !important;
    border: 1px solid var(--primary) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.15s ease !important;
    box-shadow: var(--shadow_xs) !important;
}
.input-row .stButton button:hover:not(:disabled) {
    background: var(--primary_hover) !important;
    border-color: var(--primary_hover) !important;
    transform: translateY(-1px);
    box-shadow: var(--shadow_md) !important;
}
.input-row .stButton button:active:not(:disabled) {
    background: var(--primary_active) !important;
    transform: translateY(0);
}
.input-row .stButton button:focus-visible {
    outline: 2px solid var(--primary) !important; outline-offset: 2px !important;
}
.input-row .stButton button:disabled {
    background: var(--surface_3) !important;
    color: var(--text_muted) !important;
    border-color: var(--border) !important;
    cursor: not-allowed !important;
    transform: none !important;
}
/* CHART FRAMES */
.stPlotlyChart, [data-testid="stPlotlyChart"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    box-shadow: var(--shadow_sm) !important;
    padding: 1rem !important;
    transition: box-shadow 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
}
.stPlotlyChart:hover, [data-testid="stPlotlyChart"]:hover {
    box-shadow: var(--shadow_md) !important;
    border-color: var(--border_strong) !important;
}
.js-plotly-plot, .plotly { background: transparent !important; }

/* HINT */
.hint {
    font-size: 0.82rem;
    color: var(--text_muted) !important;
    margin-top: 0.25rem;
}
.hint code {
    background: var(--surface_2);
    color: var(--text_secondary);
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.85em;
    font-family: "SF Mono", "Monaco", monospace;
    border: 1px solid var(--border);
}
.hint a { color: var(--text_secondary); text-decoration: none; }

/* DIVIDER */
.divider {
    height: 1px;
    background: var(--hairline);
    margin: 2.5rem 0;
    border: 0;
}

/* SECTION HEADER */
.section-header {
    margin-bottom: 1.5rem;
}
.section-label {
    display: block;
    font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text_muted) !important;
    margin-bottom: 0.35rem;
}
.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text_primary) !important;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.015em;
    line-height: 1.25;
}
.section-meta {
    font-size: 0.85rem;
    color: var(--text_secondary) !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.section-meta code {
    background: var(--surface_2);
    color: var(--text_primary);
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.9em;
    font-family: "SF Mono", "Monaco", monospace;
    border: 1px solid var(--border);
}

/* MODEL BADGE */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 0.5rem;
    border: 1px solid;
}
.model-badge.hybrid {
    background: var(--low_bg);
    color: var(--low) !important;
    border-color: var(--low_bg);
}
.model-badge.model {
    background: var(--primary_subtle);
    color: var(--primary) !important;
    border-color: var(--primary_border);
}
.model-badge.heuristic {
    background: var(--surface_2);
    color: var(--text_secondary) !important;
    border-color: var(--border);
}
.model-badge .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: currentColor;
}

/* STAT GRID */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.75rem;
    margin-bottom: 2.5rem;
}
.stat-cell {
    padding: 1.25rem 1.25rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    box-shadow: var(--shadow_sm);
    transition: box-shadow 0.2s ease, transform 0.2s ease, border-color 0.2s ease;
}
.stat-cell:hover {
    box-shadow: var(--shadow_md);
    transform: translateY(-1px);
    border-color: var(--border_strong);
}
.stat-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text_secondary) !important;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.stat-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text_primary) !important;
    line-height: 1;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
}
.stat-help {
    font-size: 0.72rem;
    color: var(--text_muted) !important;
    margin-top: 0.4rem;
}
.stat-cell.critical .stat-value { color: var(--critical) !important; }
.stat-cell.high .stat-value { color: var(--high) !important; }
.stat-cell.medium .stat-value { color: var(--medium) !important; }
.stat-cell.low .stat-value { color: var(--low) !important; }

/* SEVERITY BADGE */
.severity-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    line-height: 1.4;
}
.severity-badge.critical { background: var(--critical_bg); color: var(--critical) !important; }
.severity-badge.high { background: var(--high_bg); color: var(--high) !important; }
.severity-badge.medium { background: var(--medium_bg); color: var(--medium) !important; }
.severity-badge.low { background: var(--low_bg); color: var(--low) !important; }

/* RISK LIST */
.risk-list { display: flex; flex-direction: column; }
.risk-list { display: flex; flex-direction: column; gap: 0.5rem; }
.risk-row {
    display: grid;
    grid-template-columns: 48px 1fr 90px;
    gap: 1rem;
    align-items: center;
    padding: 0.9rem 1.25rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    box-shadow: var(--shadow_sm);
    transition: box-shadow 0.2s ease, transform 0.2s ease, border-color 0.2s ease;
}
.risk-row:hover {
    box-shadow: var(--shadow_md);
    transform: translateY(-1px);
    border-color: var(--border_strong);
}
.risk-row.critical { border-left: 3px solid var(--critical); }
.risk-row.high { border-left: 3px solid var(--high); }
.risk-row.medium { border-left: 3px solid var(--medium); }
.risk-row.low { border-left: 3px solid var(--low); }

.risk-rank {
    font-size: 0.85rem;
    color: var(--text_muted) !important;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    text-align: center;
}
.risk-content { min-width: 0; }
.risk-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.2rem;
    flex-wrap: wrap;
}
.risk-path {
    font-family: "SF Mono", "Monaco", "Inconsolata", monospace;
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text_primary) !important;
    word-break: break-all;
}
.risk-meta {
    font-size: 0.78rem;
    color: var(--text_secondary) !important;
    line-height: 1.4;
}
.risk-score {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text_primary) !important;
    text-align: right;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.01em;
}
.risk-row.critical .risk-score { color: var(--critical) !important; }
.risk-row.high .risk-score { color: var(--high) !important; }

/* EMPTY STATE */
.empty-state {
    padding: 3.5rem 2rem;
    text-align: center;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface);
    box-shadow: var(--shadow_sm);
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}
.empty-state::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--primary) 0%, var(--primary_hover) 100%);
}
.empty-state-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text_primary) !important;
    margin-bottom: 0.4rem;
}
.empty-state-text {
    font-size: 0.9rem;
    color: var(--text_secondary) !important;
    max-width: 380px;
    margin: 0 auto;
    line-height: 1.5;
}

/* LOADING */
.loading-card {
    padding: 2.5rem 2rem;
    text-align: center;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--surface);
    margin: 1rem 0 2rem 0;
    box-shadow: var(--shadow_md);
}
.loading-spinner {
    display: inline-block;
    width: 28px; height: 28px;
    border: 2.5px solid var(--surface_3);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: sdp-spin 0.7s linear infinite;
    margin-bottom: 1rem;
}
@keyframes sdp-spin { to { transform: rotate(360deg); } }
.loading-text {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text_primary) !important;
}
.loading-sub {
    font-size: 0.82rem;
    color: var(--text_secondary) !important;
    margin-top: 0.3rem;
}

/* ERROR */
.error-card {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    padding: 1.1rem 1.25rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--danger);
    border-radius: 10px;
    box-shadow: var(--shadow_sm);
    margin: 1rem 0;
}
.error-icon {
    color: var(--danger) !important;
    flex-shrink: 0;
    margin-top: 1px;
}
.error-icon svg { display: block; }
.error-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text_primary) !important;
    margin-bottom: 0.2rem;
}
.error-message {
    font-size: 0.85rem;
    color: var(--text_secondary) !important;
    word-break: break-word;
    line-height: 1.4;
}

/* EXPANDER */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text_primary) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    transition: background 0.15s ease, border-color 0.15s ease !important;
}
.streamlit-expanderHeader:hover {
    background: var(--surface_2) !important;
    border-color: var(--border_strong) !important;
}
.streamlit-expanderContent {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: 0 !important;
    border-radius: 0 0 8px 8px !important;
    color: var(--text_primary) !important;
}

/* SECONDARY BUTTON (Clear results) */
.secondary-btn .stButton button {
    background: var(--surface) !important;
    color: var(--text_primary) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
}
.secondary-btn .stButton button:hover:not(:disabled) {
    background: var(--surface_2) !important;
    border-color: var(--border_strong) !important;
    transform: none !important;
    box-shadow: var(--shadow_sm) !important;
}

/* SCROLLBAR */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--scrollbar_thumb); border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: var(--scrollbar_thumb_hover); }


/* THEME TOGGLE - link styled as button, JS handles click */
.topbar-theme-btn {
    position: absolute;
    top: 1.5rem;
    right: 0;
    display: inline-flex !important; align-items: center; justify-content: center;
    width: 36px; height: 36px;
    border-radius: 9px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text_secondary);
    text-decoration: none;
    transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease;
    box-shadow: var(--shadow_xs);
    cursor: pointer;
    z-index: 10;
}
.topbar-theme-btn:hover {
    background: var(--surface_2);
    border-color: var(--border_strong);
    color: var(--text_primary);
    transform: translateY(-1px);
    box-shadow: var(--shadow_sm);
}
.topbar-theme-btn:active {
    background: var(--surface_3);
    transform: translateY(0);
}
.topbar-theme-btn:focus-visible {
    outline: 2px solid var(--primary);
    outline-offset: 2px;
}
.topbar-theme-btn svg { display: block; }
/* Theme icon - specificity bumped so it beats the generic rule above.
   Sun hidden in light mode, moon hidden in dark mode. */
.topbar-theme-btn .theme-icon-sun { display: none; }
:root[data-theme="dark"] .topbar-theme-btn .theme-icon-moon { display: none; }
:root[data-theme="dark"] .topbar-theme-btn .theme-icon-sun { display: inline-flex; }

/* SMOOTH THEME TRANSITION - applies when theme flips */
html, body, .stApp, .stMarkdown, .stButton, .stTextInput, [data-baseweb] {
    transition: background-color 0.25s ease, color 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
}


/* ENTRANCE ANIMATIONS - subtle fade-in when content appears */
@keyframes sdp-fade-in {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
}
.section-header,
.stat-grid,
.risk-list,
.stPlotlyChart,
.empty-state,
.error-card,
.loading-card {
    animation: sdp-fade-in 0.35s ease-out both;
}
.risk-row:nth-child(1) { animation-delay: 0.00s; }
.risk-row:nth-child(2) { animation-delay: 0.03s; }
.risk-row:nth-child(3) { animation-delay: 0.06s; }
.risk-row:nth-child(4) { animation-delay: 0.09s; }
.risk-row:nth-child(5) { animation-delay: 0.12s; }
.risk-row:nth-child(6) { animation-delay: 0.15s; }
.risk-row:nth-child(7) { animation-delay: 0.18s; }
.risk-row:nth-child(8) { animation-delay: 0.21s; }
.risk-row:nth-child(9) { animation-delay: 0.24s; }
.risk-row:nth-child(10) { animation-delay: 0.27s; }
.risk-row:nth-child(n+11) { animation-delay: 0.30s; }

/* REMOVED OLD THEME LINK (no longer rendered, kept CSS for safety) */
.topbar-theme-btn { display: none; }

/* RESPONSIVE */
@media (max-width: 768px) {
    .main .block-container { padding-left: 1.25rem; padding-right: 1.25rem; }
    .hero { padding: 2rem 1.5rem 1.75rem 1.5rem; }
    .hero h1 { font-size: 1.85rem; }
    .hero p { font-size: 0.95rem; }
    .input-row { flex-direction: column; }
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
    .risk-row { grid-template-columns: 1fr; gap: 0.5rem; }
    .risk-score { text-align: left; }
    .risk-rank { text-align: left; }
}
</style>"""
    st.markdown(css, unsafe_allow_html=True)



def _topbar():
    # Theme toggle is positioned absolutely in the top-right corner (CSS handles placement).
    # Brand is centered and slightly larger than before for visual balance.
    moon = '<svg class="theme-icon theme-icon-moon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>'
    sun  = '<svg class="theme-icon theme-icon-sun"  width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="M4.93 4.93l1.41 1.41"/><path d="M17.66 17.66l1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="M4.93 19.07l1.41-1.41"/><path d="M17.66 6.34l1.41-1.41"/></svg>'
    st.markdown(
        f'<a class="topbar-theme-btn" data-theme-toggle title="Toggle theme" aria-label="Toggle theme">{moon}{sun}</a>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div class="topbar">
<div class="topbar-brand">
<span class="topbar-brand-mark"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></span>
<span class="topbar-brand-name">Defect Risk Analyzer</span>
<span class="topbar-brand-tag">research demo</span>
</div>
</div>""",
        unsafe_allow_html=True,
    )

def _hero():
    st.markdown(
        """<div class="hero">
<div class="hero-eyebrow">Research Demo</div>
<h1>Defect risk analysis for any GitHub repository</h1>
<p>Paste a public GitHub URL. We will download the source, score every file using software metrics and commit-message features, and surface the riskiest files with severity ratings and explanations.</p>
</div>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="input-row">', unsafe_allow_html=True)
    cols = st.columns([5, 1])
    with cols[0]:
        url = st.text_input("Repository URL", placeholder="https://github.com/owner/repo", key="gh_url_input", label_visibility="collapsed")
    with cols[1]:
        is_valid = bool(url and "github.com" in url)
        clicked = st.button("Analyze", use_container_width=True, disabled=not is_valid, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        """<div class="hint">Try: <code>facebook/react</code> &middot; <code>django/django</code> &middot; <code>pallets/flask</code> &middot; <code>requests/requests</code></div>""",
        unsafe_allow_html=True,
    )
    if clicked and is_valid:
        _run_analysis(url.strip())

def _run_analysis(url):
    st.session_state.analysis_error = None
    st.session_state.analysis_result = None
    progress = st.empty()
    progress.markdown(
        """<div class="loading-card">
<div class="loading-spinner"></div>
<div class="loading-text">Analyzing repository</div>
<div class="loading-sub">Downloading source files and scoring with the model.</div>
</div>""",
        unsafe_allow_html=True,
    )
    try:
        sig = inspect.signature(build_analysis_result)
        if len(sig.parameters) >= 3:
            r = build_analysis_result("", url, None)
        else:
            r = build_analysis_result(url)
        st.session_state.analysis_result = {
            "source": r.source,
            "file_count": r.file_count,
            "risks": [x.__dict__ for x in r.risks],
            "notes": list(r.notes) if r.notes else [],
            "excluded_files": list(r.excluded_files) if r.excluded_files else [],
            "explainability": r.explainability.__dict__ if r.explainability else None,
        }
        progress.empty()
        st.rerun()
    except Exception as e:
        progress.empty()
        st.session_state.analysis_error = str(e)
        logger.exception("Repo analysis error")
        st.rerun()



def _render_results():
    res = st.session_state.analysis_result
    risks = res.get("risks", [])
    file_count = res.get("file_count", 0)
    source = res.get("source", "")
    notes = res.get("notes", [])
    explainability = res.get("explainability")

    sev_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for r in risks:
        sev = r.get("severity", "Low")
        if sev in sev_counts:
            sev_counts[sev] += 1

    badge_html = ""
    if explainability:
        d = explainability.get("details", {})
        if d.get("hybrid_model_used"):
            badge_html = '<span class="model-badge hybrid"><span class="dot"></span>Hybrid model (metrics + commit messages)</span>'
        elif d.get("model_predictions_available"):
            badge_html = '<span class="model-badge model"><span class="dot"></span>Model-backed predictions</span>'
        else:
            badge_html = '<span class="model-badge heuristic"><span class="dot"></span>Heuristic scoring</span>'

    st.markdown('<div class="section-header">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">01 / Analysis Summary</span>', unsafe_allow_html=True)
    st.markdown(f'<h2 class="section-title">{file_count} files analyzed</h2>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-meta">Source <code>{source}</code></div>', unsafe_allow_html=True)
    if badge_html:
        st.markdown(badge_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="stat-grid">
<div class="stat-cell"><div class="stat-label">Files</div><div class="stat-value">{file_count}</div><div class="stat-help">in repository</div></div>
<div class="stat-cell critical"><div class="stat-label">Critical</div><div class="stat-value">{sev_counts.get("Critical", 0)}</div><div class="stat-help">90% risk or higher</div></div>
<div class="stat-cell high"><div class="stat-label">High</div><div class="stat-value">{sev_counts.get("High", 0)}</div><div class="stat-help">75 to 90% risk</div></div>
<div class="stat-cell medium"><div class="stat-label">Medium</div><div class="stat-value">{sev_counts.get("Medium", 0)}</div><div class="stat-help">50 to 75% risk</div></div>
<div class="stat-cell low"><div class="stat-label">Low</div><div class="stat-value">{sev_counts.get("Low", 0)}</div><div class="stat-help">below 50% risk</div></div>
</div>""", unsafe_allow_html=True)

    if risks and go is not None:
        st.markdown('<div class="section-header">', unsafe_allow_html=True)
        st.markdown('<span class="section-label">02 / Distribution</span>', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">How risk is distributed</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            f = _severity_donut(sev_counts)
            if f: st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})
        with c2:
            f = _top_files_bar(risks, n=10)
            if f: st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            f = _risk_distribution(risks)
            if f: st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})
        with c2:
            f = _severity_by_extension(risks)
            if f: st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})

    if risks:
        st.markdown('<div class="section-header">', unsafe_allow_html=True)
        st.markdown('<span class="section-label">03 / Risk Files</span>', unsafe_allow_html=True)
        st.markdown(f'<h2 class="section-title">Top {min(15, len(risks))} highest-risk files</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        rows = []
        for i, r in enumerate(risks[:15], 1):
            sev = str(r.get("severity", "Low")).lower()
            path = r.get("path", "?")
            prob = r.get("probability", "?")
            reason = r.get("reason", "") or ""
            rows.append(
                f'<div class="risk-row {sev}">'
                f'<div class="risk-rank">{i:02d}</div>'
                f'<div class="risk-content">'
                f'<div class="risk-header">'
                f'<span class="severity-badge {sev}">{sev}</span>'
                f'<span class="risk-path">{path}</span>'
                f'</div>'
                f'<div class="risk-meta">{reason}</div>'
                f'</div>'
                f'<div class="risk-score">{prob}</div>'
                f'</div>'
            )
        st.markdown('<div class="risk-list">' + "".join(rows) + '</div>', unsafe_allow_html=True)
        st.markdown("", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 4])
        with c1:
            st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
            if st.button("Clear results", use_container_width=True):
                st.session_state.analysis_result = None
                st.session_state.analysis_error = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    if notes:
        with st.expander(f"Analysis notes ({len(notes)})"):
            for n in notes:
                st.markdown(f"- {n}")
    excluded = res.get("excluded_files", [])
    if excluded:
        with st.expander(f"Excluded files ({len(excluded)})"):
            for ef in excluded:
                st.markdown(f"- `{ef}`")



def _empty_state():
    st.markdown(
        """<div class="empty-state">
<div class="empty-state-title">No analysis yet</div>
<div class="empty-state-text">Submit a public GitHub repository URL above to begin. Results will appear here within a few seconds.</div>
</div>""",
        unsafe_allow_html=True,
    )

def _error_state():
    err = st.session_state.analysis_error
    st.markdown(
        f"""<div class="error-card">
<div class="error-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></div>
<div>
<div class="error-title">Analysis failed</div>
<div class="error-message">{err}</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )
    if st.button("Try again", type="primary"):
        st.session_state.analysis_error = None
        st.rerun()

def _severity_donut(sev_counts):
    if go is None: return None
    t = _t()
    labels, values, colors = [], [], []
    cm = {"Critical": t["critical"], "High": t["high"], "Medium": t["medium"], "Low": t["low"]}
    for sev in ["Critical", "High", "Medium", "Low"]:
        c = sev_counts.get(sev, 0)
        if c > 0:
            labels.append(sev); values.append(c); colors.append(cm[sev])
    if not values: return None
    total = sum(values)
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.65,
        marker=dict(colors=colors, line=dict(color=t["surface"], width=2)),
        textinfo="label+percent", textposition="outside", sort=False,
        textfont=dict(size=12, color=t["text_secondary"]),
    )])
    fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color=t["text_primary"]),
        title=dict(text="Severity distribution", x=0.05, font_size=14, font_color=t["text_primary"]),
        annotations=[dict(text=f"{total}<br>files", x=0.5, y=0.5, font_size=14, showarrow=False, font_color=t["text_secondary"])],
        showlegend=False,
    )
    return fig

def _top_files_bar(risks, n=10):
    if go is None or not risks: return None
    t = _t()
    rows = []
    for r in risks:
        try: prob = float(str(r.get("probability", "0%")).replace("%", "")) / 100.0
        except (TypeError, ValueError): prob = 0.0
        rows.append({"path": r.get("path", "?"), "prob": prob, "sev": r.get("severity", "Low")})
    rows = sorted(rows, key=lambda x: x["prob"], reverse=True)[:n]
    if not rows: return None
    cm = {"Critical": t["critical"], "High": t["high"], "Medium": t["medium"], "Low": t["low"]}
    paths = [r["path"] for r in rows]
    probs = [r["prob"] for r in rows]
    colors = [cm.get(r["sev"], t["text_muted"]) for r in rows]
    dp = [p if len(p) <= 45 else "\u2026" + p[-42:] for p in paths]
    fig = go.Figure(go.Bar(
        x=probs, y=dp, orientation="h",
        marker=dict(color=colors, line_width=0),
        text=[f"{int(p*100)}%" for p in probs], textposition="outside", cliponaxis=False,
        textfont=dict(size=11, color=t["text_secondary"]),
        hovertemplate="<b>%{y}</b><br>Risk: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(
        height=max(260, len(rows) * 26 + 50), margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(title="Defect risk", range=[0, 1.1], gridcolor="rgba(128,128,128,0.12)", tickformat=".0%", title_font=dict(size=12, color=t["text_primary"])),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color=t["text_primary"])),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color=t["text_primary"]),
        title=dict(text="Top risk files", x=0.05, font_size=14, font_color=t["text_primary"]),
    )
    return fig

def _risk_distribution(risks):
    if go is None or not risks: return None
    t = _t()
    probs = []
    for r in risks:
        try: probs.append(float(str(r.get("probability", "0%")).replace("%", "")) / 100.0)
        except (TypeError, ValueError): pass
    if not probs: return None
    fig = go.Figure(go.Histogram(
        x=probs, nbinsx=15,
        marker=dict(color=t["primary"], line=dict(color=t["surface"], width=1)),
        hovertemplate="Risk: %{x}<br>Files: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0.9, line_dash="dash", line_color=t["critical"], annotation_text="Critical 90%", annotation_position="top right")
    fig.add_vline(x=0.75, line_dash="dash", line_color=t["high"], annotation_text="High 75%", annotation_position="top right")
    fig.add_vline(x=0.5, line_dash="dash", line_color=t["medium"], annotation_text="Medium 50%", annotation_position="top right")
    fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(title="Defect risk probability", tickformat=".0%", gridcolor="rgba(128,128,128,0.12)", title_font=dict(size=12, color=t["text_primary"])),
        yaxis=dict(title="Number of files", gridcolor="rgba(128,128,128,0.12)", title_font=dict(size=12, color=t["text_primary"])),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color=t["text_primary"]),
        title=dict(text="Risk score distribution", x=0.05, font_size=14, font_color=t["text_primary"]),
        showlegend=False,
    )
    return fig

def _severity_by_extension(risks):
    if go is None or not risks: return None
    t = _t()
    from pathlib import PurePosixPath
    rows = []
    for r in risks:
        path = r.get("path", "")
        ext = PurePosixPath(path).suffix.lower() or "(no ext)"
        rows.append({"ext": ext, "sev": r.get("severity", "Low")})
    if not rows: return None
    df = pd.DataFrame(rows)
    sev_set = ["Critical", "High", "Medium", "Low"]
    cm = {"Critical": t["critical"], "High": t["high"], "Medium": t["medium"], "Low": t["low"]}
    ext_totals = df["ext"].value_counts().head(8).index.tolist()
    df = df[df["ext"].isin(ext_totals)]
    fig = go.Figure()
    for sev in sev_set:
        sub = df[df["sev"] == sev].groupby("ext").size().reindex(ext_totals, fill_value=0)
        fig.add_trace(go.Bar(
            x=sub.index, y=sub.values,
            marker=dict(color=cm[sev], line_width=0),
            name=sev,
        ))
    fig.update_layout(
        barmode="stack",
        height=280,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(title="File count", gridcolor="rgba(128,128,128,0.12)", title_font=dict(size=12, color=t["text_primary"])),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color=t["text_primary"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        title=dict(text="Severity by file extension", x=0.05, font_size=14, font_color=t["text_primary"]),
        xaxis=dict(tickangle=-25, tickfont=dict(size=11, color=t["text_primary"])),
    )
    return fig

def main():
    # Theme is handled entirely client-side via JS toggling data-theme on <html>.
    # No Streamlit rerun, no page reload - the switch is instant.
    _inject_css()
    _theme_js()
    _topbar()
    _hero()
    if st.session_state.get("analysis_result"):
        _render_results()
    elif st.session_state.get("analysis_error"):
        _error_state()
    else:
        _empty_state()

if __name__ == "__main__":
    main()
