
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from models.ml_engine import run_ml_pipeline

st.header("4. Machine Learning Pipeline")

if st.session_state.get('clean_data') is not None:
    df = st.session_state['clean_data']
    
    target = st.selectbox("Selecciona la variable Objetivo (Target)", df.columns)
    tarea = st.radio("Tipo de Tarea", ["Clasificación", "Regresión"])
    
    if st.button("Entrenar Modelo (Random Forest)"):
        with st.spinner("Entrenando modelo..."):
            try:
                model, metrics = run_ml_pipeline(df, target, tarea)
                st.success("¡Entrenamiento finalizado!")
                st.json(metrics)
            except Exception as e:
                st.error(f"Error en el entrenamiento. Verifica que las características estén limpias y codificadas. Detalles: {e}")
else:
    st.warning("Por favor, carga un dataset en el paso 1.")