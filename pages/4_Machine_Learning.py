import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import io
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
                st.session_state['ml_model'] = model
                st.session_state['ml_metrics'] = metrics
                st.success("¡Entrenamiento finalizado!")
            except Exception as e:
                st.error(f"Error en el entrenamiento. Verifica que las características estén limpias y codificadas. Detalles: {e}")

    if 'ml_model' in st.session_state:
        st.subheader("Resultados y Exportación")
        st.json(st.session_state['ml_metrics'])
        
        # Preparar el modelo para descarga mediante serialización
        model_buffer = io.BytesIO()
        pickle.dump(st.session_state['ml_model'], model_buffer)
        
        st.download_button(
            label="📥 Descargar Modelo Random Forest (.pkl)",
            data=model_buffer.getvalue(),
            file_name=f"modelo_rf_{tarea.lower()}.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )
else:
    st.warning("Por favor, carga un dataset en el paso 1.")