import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
from charts.visualizer import plot_correlation_matrix, plot_distribution

st.header("3. Visualización y Análisis Estadístico")

if st.session_state.get('clean_data') is not None:
    df = st.session_state['clean_data']

    tab1, tab2 = st.tabs(["Distribuciones", "Correlaciones"])

    # Debug opcional para ver estado de la sesión y estructura del dataframe
    with st.expander("🔍 Debug (mostrar estado y estructura)", expanded=False):
        st.write("`st.session_state` keys:", list(st.session_state.keys()))
        st.write("Columnas:", df.columns.tolist())
        st.write("Tipos:")
        st.write(df.dtypes)
        st.write("Primeras filas:")
        st.dataframe(df.head(5))

    with tab1:
        cols_num = df.select_dtypes(include=['number']).columns.tolist()
        if not cols_num:
            st.info("No se encontraron columnas numéricas para visualizar. Revisa el dataset o aplica transformaciones en Limpieza.")
        else:
            col_dist = st.selectbox("Selecciona variable numérica", cols_num)
            if col_dist:
                try:
                    fig = plot_distribution(df, col_dist)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error al generar la distribución: {e}")

    with tab2:
        st.write("Matriz de Correlación de Pearson")
        try:
            fig_corr = plot_correlation_matrix(df)
            st.pyplot(fig_corr)
        except Exception as e:
            st.error(f"Error al generar la matriz de correlación: {e}")
else:
    st.warning("Por favor, carga un dataset en el paso 1.")