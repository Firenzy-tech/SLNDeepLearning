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


def get_cleaning_recommendation(dtype: str, null_count: int, total_rows: int, col_name: str) -> str:
    """
    Genera recomendaciones de limpieza según el tipo de dato y características.
    Sigue estándares de ciencia de datos y buenas prácticas.
    """
    null_percentage = (null_count / total_rows * 100) if total_rows > 0 else 0
    
    # Recomendaciones por tipo de dato
    if 'int' in dtype or 'float' in dtype:
        if null_count > 0:
            if null_percentage < 5:
                return f"✅ Imputar con media/mediana (nulos: {null_percentage:.1f}%)"
            elif null_percentage < 20:
                return f"⚠️ Imputar o eliminar filas (nulos: {null_percentage:.1f}%)"
            else:
                return f"❌ Considerar eliminar columna (nulos: {null_percentage:.1f}%)"
        return "✅ Usar escaling (Min-Max o Standardización)"
    
    elif 'object' in dtype or 'string' in dtype:
        if null_count > 0:
            if null_percentage < 5:
                return f"✅ Usar moda o crear categoría 'Unknown' (nulos: {null_percentage:.1f}%)"
            elif null_percentage < 20:
                return f"⚠️ Evaluar imputación o eliminar (nulos: {null_percentage:.1f}%)"
            else:
                return f"❌ Considerar eliminar columna (nulos: {null_percentage:.1f}%)"
        return "✅ Codificar (Label Encoding o One-Hot)"
    
    elif 'datetime' in dtype or 'date' in dtype:
        if null_count > 0:
            return f"✅ Eliminar o interpolar temporalmente (nulos: {null_percentage:.1f}%)"
        return "✅ Extraer features (año, mes, día, día semana)"
    
    elif 'bool' in dtype or 'category' in dtype:
        if null_count > 0:
            return f"⚠️ Imputar con moda (nulos: {null_percentage:.1f}%)"
        return "✅ Codificar como 0/1 o usar One-Hot si hay múltiples categorías"
    
    else:
        return "⚠️ Revisar tipo de dato - considerar conversión"


def get_data_types_with_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna un DataFrame con recomendaciones de limpieza para cada columna.
    Incluye: Campo, Tipo, Valores Nulos, % Nulos, Recomendación
    """
    total_rows = len(df)
    data_info = []
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_percentage = (null_count / total_rows * 100) if total_rows > 0 else 0
        dtype = str(df[col].dtype)
        recommendation = get_cleaning_recommendation(dtype, null_count, total_rows, col)
        
        data_info.append({
            "📋 Campo": col,
            "🔢 Tipo": dtype,
            "❌ Nulos": null_count,
            "% Nulos": f"{null_percentage:.1f}%",
            "💡 Recomendación": recommendation
        })
    
    return pd.DataFrame(data_info)