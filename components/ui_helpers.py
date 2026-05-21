
#Componentes UI reutilizables para mantener la consistencia visual.
import streamlit as st
from typing import Any

def render_kpi_card(title: str, value: Any, delta: str = None):
    """Renderiza una tarjeta de métrica KPI estandarizada."""
    st.metric(label=title, value=value, delta=delta)

import contextlib
import base64
import os

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
    <div style="display: flex; align-items: center; gap: 10px; padding: 10px; font-family: sans-serif;">
        <img src="data:image/svg+xml;base64,{spinner_b64}" width="30" height="30" style="animation: spin_custom 1s linear infinite;" />
        <span style="font-size: 14px;">{message}</span>
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

def setup_branding():
    """Configura el logo y el estilo global del spinner."""
    import os
    import base64
    logo_path = os.path.join(os.path.dirname(__file__), "..", "wwwroot", "images", "golondrina_software.svg")
    
    logo_b64 = ""
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
            
    st.markdown(f"""
    <style>
    /* Ocultar el logo nativo restrictivo de Streamlit si existe */
    [data-testid="stLogo"] {{
        display: none !important;
    }}
    [data-testid="stSidebarHeader"] {{
        display: none !important;
    }}

    /* Títulos principales con gradiente */
    h1 {{
        background: -webkit-linear-gradient(45deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Inyectar el logo en el header de la página principal
    if logo_b64:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: flex-start; margin-bottom: 1rem;">
                <img src="data:image/svg+xml;base64,{logo_b64}" style="height: 100px; max-width: 100%; object-fit: contain;" />
            </div>
            """,
            unsafe_allow_html=True
        )