
#Componentes UI reutilizables para mantener la consistencia visual.
import streamlit as st
from typing import Any

def render_kpi_card(title: str, value: Any, delta: str = None):
    """Renderiza una tarjeta de métrica KPI estandarizada."""
    st.metric(label=title, value=value, delta=delta)

def show_spinner(message: str = "Procesando..."):
    """Wrapper para el spinner de Streamlit."""
    return st.spinner(message)