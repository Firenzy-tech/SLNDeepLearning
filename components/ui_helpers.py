# Componentes UI reutilizables para mantener la consistencia visual.
from __future__ import annotations

import base64
import contextlib
import os
from typing import Any

import streamlit as st


_GLOBAL_STYLE = """
<style>
:root {
    --ui-primary: #022873;
    --ui-secondary: #021F59;
    --ui-surface: #B4C4D9;
    --ui-accent: #D97904;
    --ui-danger: #BF2604;
    --ui-background: #F4F7FC;
    --ui-surface-strong: #FFFFFF;
    --ui-surface-muted: #E4ECF7;
    --ui-border: rgba(2, 31, 89, 0.14);
    --ui-border-strong: rgba(2, 40, 115, 0.30);
    --ui-text: #021F59;
    --ui-text-soft: rgba(2, 31, 89, 0.78);
    --ui-shadow: 0 18px 48px rgba(2, 40, 115, 0.12);
    --ui-shadow-soft: 0 8px 24px rgba(2, 40, 115, 0.08);
    --ui-radius-sm: 0.65rem;
    --ui-radius-md: 0.95rem;
    --ui-radius-lg: 1.25rem;
    --ui-focus: 0 0 0 3px rgba(217, 121, 4, 0.22);
    --ui-transition: 160ms ease;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(2, 40, 115, 0.14), transparent 34%),
        radial-gradient(circle at top right, rgba(217, 121, 4, 0.08), transparent 26%),
        linear-gradient(180deg, #F8FBFF 0%, #EEF3FA 46%, #E6EDF7 100%);
    color: var(--ui-text);
}

[data-testid="stToolbar"] {
    visibility: visible;
    height: auto;
    background: transparent;
    z-index: 1000;
}

[data-testid="stHeader"] {
    display: block !important;
    position: fixed;
    inset: 0 0 auto 0;
    height: 0;
    min-height: 0;
    overflow: visible;
    background: transparent;
    border: 0;
    box-shadow: none;
    opacity: 1;
    z-index: 1001;
    pointer-events: none;
}

[data-testid="stHeader"] * {
    pointer-events: auto;
}

[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[aria-label*="Open sidebar"] {
    position: fixed !important;
    top: 0.7rem;
    left: 0.7rem;
    display: inline-flex !important;
    align-items: center;
    justify-content: center;
    width: 2.25rem;
    height: 2.25rem;
    border-radius: 0.7rem;
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid rgba(2, 40, 115, 0.18);
    box-shadow: 0 10px 24px rgba(2, 40, 115, 0.14);
    color: var(--ui-primary) !important;
    z-index: 2000;
}

[data-testid="collapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg,
[aria-label*="Open sidebar"] svg {
    color: var(--ui-primary) !important;
    fill: var(--ui-primary) !important;
}

.block-container {
    padding-top: 0.9rem;
    padding-bottom: 2.5rem;
    max-width: 1280px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #021F59 0%, #022873 100%);
    border-right: 1px solid rgba(180, 196, 217, 0.18);
}

@media (min-width: 768px) {
    [data-testid="stSidebar"] {
        overflow: hidden;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        display: block !important;
        opacity: 1;
        transform: translateX(0);
        transition: opacity 180ms ease, transform 180ms ease;
    }

    [data-testid="stSidebar"]:not(:hover) [data-testid="stSidebarContent"] {
        opacity: 1;
        transform: none;
    }
}

[data-testid="stSidebar"] * {
    color: #F5F8FC;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: rgba(245, 248, 252, 0.92) !important;
}

[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] [data-baseweb="button"] button {
    background: rgba(255, 255, 255, 0.08);
    color: #FFFFFF;
    border: 1px solid rgba(255, 255, 255, 0.16);
    box-shadow: none;
}

[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] button:hover,
[data-testid="stSidebar"] [data-baseweb="button"] button:hover {
    background: rgba(255, 255, 255, 0.14);
    border-color: rgba(255, 255, 255, 0.26);
}

h1, h2, h3, h4, h5, h6 {
    color: var(--ui-secondary);
    letter-spacing: -0.02em;
}

h1 {
    line-height: 1.08;
}

p, li, span, label, .stCaption {
    color: var(--ui-text-soft);
}

a {
    color: var(--ui-primary);
}

.stButton > button,
button[kind="secondary"],
button[kind="tertiary"],
[data-baseweb="button"] button {
    min-height: 2.75rem;
    border-radius: 0.9rem;
    border: 1px solid rgba(2, 40, 115, 0.18);
    background: linear-gradient(180deg, #FFFFFF 0%, #F2F6FB 100%);
    color: var(--ui-primary);
    font-weight: 700;
    box-shadow: var(--ui-shadow-soft);
    transition: transform var(--ui-transition), box-shadow var(--ui-transition), border-color var(--ui-transition), background var(--ui-transition);
}

.stButton > button:hover,
button[kind="secondary"]:hover,
button[kind="tertiary"]:hover,
[data-baseweb="button"] button:hover {
    transform: translateY(-1px);
    border-color: rgba(2, 40, 115, 0.32);
    box-shadow: 0 14px 32px rgba(2, 40, 115, 0.14);
}

.stButton > button:focus-visible,
button[kind="secondary"]:focus-visible,
button[kind="tertiary"]:focus-visible,
[data-baseweb="button"] button:focus-visible {
    outline: none;
    box-shadow: var(--ui-focus), var(--ui-shadow-soft);
}

.stButton > button[kind="primary"],
button[kind="primary"],
[data-baseweb="button"] button[kind="primary"] {
    background: linear-gradient(135deg, var(--ui-accent), #B96403);
    color: #FFFFFF;
    border-color: rgba(217, 121, 4, 0.32);
}

.stButton > button[kind="primary"]:hover,
button[kind="primary"]:hover,
[data-baseweb="button"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #E1830D, var(--ui-accent));
    border-color: rgba(217, 121, 4, 0.48);
}

[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea,
[data-baseweb="select"] > div,
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
.stDateInput input {
    background: #FFFFFF;
    color: var(--ui-text);
    border: 1px solid rgba(2, 40, 115, 0.18) !important;
    border-radius: 0.85rem !important;
    box-shadow: inset 0 1px 1px rgba(2, 31, 89, 0.03);
    transition: border-color var(--ui-transition), box-shadow var(--ui-transition), transform var(--ui-transition);
}

[data-baseweb="input"] input:hover,
[data-baseweb="textarea"] textarea:hover,
[data-baseweb="select"] > div:hover,
.stTextInput input:hover,
.stTextArea textarea:hover,
.stNumberInput input:hover,
.stDateInput input:hover {
    border-color: rgba(2, 40, 115, 0.32) !important;
}

[data-baseweb="input"] input:focus,
[data-baseweb="textarea"] textarea:focus,
[data-baseweb="select"] > div:focus-within,
.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus,
.stDateInput input:focus {
    border-color: var(--ui-primary) !important;
    box-shadow: var(--ui-focus) !important;
}

label {
    color: var(--ui-secondary) !important;
    font-weight: 650 !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.96) 0%, rgba(244, 248, 253, 0.96) 100%);
    border: 1px solid var(--ui-border);
    border-radius: 1rem;
    padding: 1rem 1.1rem;
    box-shadow: var(--ui-shadow-soft);
}

[data-testid="stMetricLabel"] {
    color: var(--ui-text-soft) !important;
    font-size: 0.88rem;
}

[data-testid="stMetricValue"] {
    color: var(--ui-primary) !important;
    font-weight: 800;
}

[data-testid="stMetricDelta"] {
    color: var(--ui-accent) !important;
}

[data-testid="stDataFrame"],
[data-testid="stTable"],
.stDataFrame {
    background: #FFFFFF;
    border: 1px solid var(--ui-border);
    border-radius: 1rem;
    overflow: hidden;
    box-shadow: var(--ui-shadow-soft);
}

[data-testid="stDataFrame"] [role="grid"],
[data-testid="stTable"] table {
    background: #FFFFFF;
}

[data-testid="stAlert"] {
    border-radius: 0.95rem;
    border: 1px solid rgba(2, 40, 115, 0.16);
    box-shadow: var(--ui-shadow-soft);
    background: rgba(244, 248, 253, 0.98);
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] div,
[data-testid="stAlert"] span {
    color: var(--ui-text) !important;
}

[data-testid="stAlert"] svg {
    color: var(--ui-primary);
}

button[data-baseweb="tab"] {
    background: transparent;
    color: var(--ui-text-soft);
    border: none;
    border-bottom: 2px solid transparent;
    box-shadow: none;
    font-weight: 700;
}

button[data-baseweb="tab"]:hover {
    color: var(--ui-primary);
    transform: none;
    box-shadow: none;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--ui-primary);
    border-bottom-color: var(--ui-accent);
    background: rgba(2, 40, 115, 0.05);
}

details {
    border: 1px solid var(--ui-border);
    border-radius: 1rem;
    background: rgba(255, 255, 255, 0.8);
    box-shadow: var(--ui-shadow-soft);
}

details summary {
    color: var(--ui-secondary);
    font-weight: 700;
}

hr {
    border-color: rgba(2, 40, 115, 0.14);
}

#MainMenu, footer {
    visibility: hidden;
}

.ui-brand-bar {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    padding: 0.95rem 1.15rem;
    margin-bottom: 1rem;
    border-radius: 1.1rem;
    background: linear-gradient(135deg, rgba(2, 40, 115, 0.08), rgba(180, 196, 217, 0.24));
    border: 1px solid rgba(2, 40, 115, 0.12);
}

.ui-brand-bar__logo {
    width: 44px;
    height: 44px;
    object-fit: contain;
    flex: 0 0 auto;
}

.ui-brand-bar__eyebrow {
    color: var(--ui-accent);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.72rem;
    font-weight: 800;
    margin: 0;
}

.ui-brand-bar__title {
    color: var(--ui-secondary);
    font-size: 1rem;
    font-weight: 800;
    margin: 0.05rem 0 0;
}

.ui-brand-bar__subtitle {
    color: var(--ui-text-soft);
    font-size: 0.9rem;
    margin: 0.15rem 0 0;
}
</style>
"""


def render_kpi_card(title: str, value: Any, delta: str = None):
    """Renderiza una tarjeta de métrica KPI estandarizada."""
    st.metric(label=title, value=value, delta=delta)


@contextlib.contextmanager
def show_spinner(message: str = "Procesando..."):
    """Wrapper para el spinner de Streamlit usando spinner.svg de wwwroot."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    spinner_path = os.path.join(base_dir, "wwwroot", "images", "spinner.svg")

    spinner_b64 = ""
    if os.path.exists(spinner_path):
        with open(spinner_path, "rb") as f:
            spinner_b64 = base64.b64encode(f.read()).decode()

    if not spinner_b64:
        with st.spinner(message):
            yield
        return

    placeholder = st.empty()
    html = f"""
    <div style="display:flex;align-items:center;gap:12px;padding:12px 14px;border-radius:14px;border:1px solid rgba(2,40,115,0.12);background:rgba(255,255,255,0.92);box-shadow:0 10px 22px rgba(2,40,115,0.08);">
        <img src="data:image/svg+xml;base64,{spinner_b64}" width="30" height="30" style="animation: spin_custom 1s linear infinite;" />
        <span style="font-size:14px;font-weight:600;color:#021F59;">{message}</span>
    </div>
    <style>
    @keyframes spin_custom {{ 100% {{ transform: rotate(360deg); }} }}
    </style>
    """
    placeholder.markdown(html, unsafe_allow_html=True)
    try:
        yield
    finally:
        placeholder.empty()


def setup_branding(page_title: str | None = None, page_subtitle: str | None = None, show_logo: bool = True):
    """Aplica el sistema visual corporativo global y, opcionalmente, un banner de marca."""
    logo_path = os.path.join(os.path.dirname(__file__), "..", "wwwroot", "images", "golondrina_software.svg")

    logo_b64 = ""
    if show_logo and os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()

    st.markdown(_GLOBAL_STYLE, unsafe_allow_html=True)

    if show_logo and logo_b64:
        subtitle_html = f'<p class="ui-brand-bar__subtitle">{page_subtitle}</p>' if page_subtitle else ""
        title_html = f'<p class="ui-brand-bar__title">{page_title}</p>' if page_title else '<p class="ui-brand-bar__title">Plataforma corporativa de analítica</p>'
        st.markdown(
            f"""
            <div class="ui-brand-bar">
                <img class="ui-brand-bar__logo" src="data:image/svg+xml;base64,{logo_b64}" alt="Logo de la plataforma" />
                <div>
                    <p class="ui-brand-bar__eyebrow">Data Science &amp; ML</p>
                    {title_html}
                    {subtitle_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )