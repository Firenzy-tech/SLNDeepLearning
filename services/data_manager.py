import pandas as pd
import streamlit as st
from typing import Tuple, Dict, Any
import io

@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    """
    Carga el dataset dependiendo de su extensión.
    Se utiliza @st.cache_data para evitar recargas en cada interacción de la UI.
    """
    try:
        filename = uploaded_file.name
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        elif filename.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            raise ValueError("Formato no soportado")
        return df
    except Exception as e:
        raise RuntimeError(f"Error al cargar el archivo: {str(e)}")

def get_profile_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Genera estadísticas descriptivas para el perfilado del dataset."""
    stats = {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing_cells": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum(),
        "memory_usage": df.memory_usage(deep=True).sum() / 1024**2 # en MB
    }
    return stats