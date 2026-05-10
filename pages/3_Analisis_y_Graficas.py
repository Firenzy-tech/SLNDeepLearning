import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from charts.visualizer import plot_correlation_matrix, plot_distribution

st.header("3. Visualización y Análisis Estadístico")

if st.session_state.get('clean_data') is not None:
    df = st.session_state['clean_data']
    
    tab1, tab2 = st.tabs(["Distribuciones", "Correlaciones"])
    
    with tab1:
        cols_num = df.select_dtypes(include=['number']).columns.tolist()
        col_dist = st.selectbox("Selecciona variable numérica", cols_num)
        if col_dist:
            fig = plot_distribution(df, col_dist)
            st.pyplot(fig)
            
    with tab2:
        st.write("Matriz de Correlación de Pearson")
        fig_corr = plot_correlation_matrix(df)
        st.pyplot(fig_corr)
else:
    st.warning("Por favor, carga un dataset en el paso 1.")