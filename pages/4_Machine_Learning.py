import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

        # Nueva sección para inspeccionar la estructura interna del archivo PKL
        with st.expander("🔍 Inspección del Modelo (Contenido del PKL)", expanded=False):
            model = st.session_state['ml_model']
            
            st.markdown("### ⚙️ Hiperparámetros del Algoritmo")
            st.write("Esta es la configuración técnica que quedará grabada en el archivo .pkl:")
            st.json(model.get_params())
            
            if hasattr(model, 'feature_importances_'):
                st.markdown("### 📊 Relevancia de Variables (Feature Importance)")
                # Intentamos obtener las columnas del modelo si es un Pipeline, 
                # de lo contrario usamos las del dataframe actual excluyendo el target
                features = getattr(model, 'feature_names_in_', [c for c in df.columns if c != target])
                importances = model.feature_importances_
                
                if len(features) == len(importances):
                    feat_imp = pd.DataFrame({'Variable': features, 'Importancia': importances}).sort_values(by='Importancia', ascending=False)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    # Usamos una paleta de colores para mejor visualización
                    sns.barplot(data=feat_imp.head(12), x='Importancia', y='Variable', ax=ax, palette='viridis')
                    ax.set_title("Top Variables que influyen en el Modelo")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("⚠️ No se puede graficar la importancia: El modelo tiene dimensiones distintas al dataset original (posiblemente por One-Hot Encoding interno).")
        
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