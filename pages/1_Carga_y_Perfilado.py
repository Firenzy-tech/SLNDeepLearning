import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from services.data_manager import load_data, get_profile_stats, get_data_types_with_recommendations
from components.ui_helpers import render_kpi_card, show_spinner

st.header("1. Carga de Archivos y Perfilado")

uploaded_file = st.file_uploader("Sube tu dataset (CSV, Excel, JSON)", type=['csv', 'xlsx', 'xls', 'json'])

max_rows = None
if uploaded_file is not None:
    # Mover las opciones de control antes de la carga efectiva
    with st.expander("⚙️ Opciones de Carga Avanzadas", expanded=True):
        # Por defecto NO limitar filas; el usuario puede activar la limitación si lo desea.
        limit_rows = st.checkbox("Limitar filas para no saturar la RAM (recomendado)", value=False)
        if limit_rows:
            max_rows = st.number_input("Cantidad de filas", min_value=1, value=10000, step=1000)

if uploaded_file is not None:
    with show_spinner("Cargando datos en memoria..."):
        # Pasamos un file_id basado en nombre y tamaño para que el caché sea rápido
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        df = load_data(uploaded_file, max_rows=max_rows, file_id=file_id)
        
        st.session_state['raw_data'] = df
        # Optimizamos: No hacemos .copy() inmediatamente si el archivo es grande
        # Si `clean_data` aún es None, lo inicializamos con los datos cargados
        if st.session_state.get('clean_data') is None:
            st.session_state['clean_data'] = df
        
    st.success("¡Datos cargados exitosamente!")
    
    # Tarjetas KPI
    stats = get_profile_stats(df)
    col1, col2, col3, col4 = st.columns(4)
    with col1: render_kpi_card("Filas", stats['rows'])
    with col2: render_kpi_card("Columnas", stats['cols'])
    with col3: render_kpi_card("Valores Nulos", stats['missing_cells'])
    with col4: render_kpi_card("Memoria (MB)", f"{stats['memory_usage']:.2f}")
    
    st.subheader("Vista Previa")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("📊 Análisis de Tipos de Datos y Recomendaciones")
    st.markdown("""
    Tabla con análisis automático de cada columna, incluyendo tipo de dato, cantidad de valores nulos 
    y recomendaciones de limpieza según estándares de ciencia de datos.
    """)
    
    data_types_df = get_data_types_with_recommendations(df)
    st.dataframe(data_types_df, use_container_width=True, hide_index=True)
    
    # Resumen de alertas
    st.divider()
    st.subheader("⚠️ Alertas y Acciones Recomendadas")
    
    cols_with_nulls = data_types_df[data_types_df["❌ Nulos"] > 0]
    if len(cols_with_nulls) > 0:
        st.warning(f"🔴 {len(cols_with_nulls)} columna(s) con valores nulos detectadas")
        for idx, row in cols_with_nulls.iterrows():
            st.write(f"  • **{row['📋 Campo']}**: {row['❌ Nulos']} nulos ({row['% Nulos']}) - {row['💡 Recomendación']}")
    else:
        st.success("✅ No se detectaron valores nulos")