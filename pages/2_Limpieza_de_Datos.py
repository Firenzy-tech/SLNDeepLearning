import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
try:
    from services.cleaner_service import DataCleaner
    CLEANER_AVAILABLE = True
    CLEANER_IMPORT_ERROR = None
except Exception as _e:
    DataCleaner = None
    CLEANER_AVAILABLE = False
    CLEANER_IMPORT_ERROR = _e

st.header("2. Limpieza y Transformación")

if st.session_state.get('clean_data') is not None:
    df = st.session_state['clean_data']
    
    st.subheader("Manejo de Valores Nulos")
    cols_con_nulos = df.columns[df.isnull().any()].tolist()
    
    if not CLEANER_AVAILABLE:
        st.error(f"Error importing cleaning service: {CLEANER_IMPORT_ERROR}.\nInstala dependencias: pip install -r requirements.txt")
    elif cols_con_nulos:
        col_selec = st.multiselect("Columnas a imputar/limpiar", cols_con_nulos)
        estrategia = st.selectbox("Estrategia", ["Eliminar filas", "Imputar media", "Imputar mediana", "Forward fill"])
        
        if st.button("Aplicar Limpieza de Nulos"):
            df = DataCleaner.handle_missing(df, estrategia, col_selec)
            st.session_state['clean_data'] = df
            st.success("Limpieza aplicada. Estado actualizado.")
            st.rerun()
    else:
        st.info("No se detectaron valores nulos en el dataset.")

    st.divider()
    
    st.subheader("Codificación Categórica")
    cols_cat = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cols_cat:
        cat_col_selec = st.multiselect("Columnas a codificar", cols_cat)
        metodo_cod = st.selectbox("Método", ["Label Encoding", "One Hot Encoding"])
        
        if st.button("Aplicar Codificación"):
            if not CLEANER_AVAILABLE:
                st.error(f"Error importing cleaning service: {CLEANER_IMPORT_ERROR}.\nInstala dependencias: pip install -r requirements.txt")
            else:
                df = DataCleaner.encode_categorical(df, cat_col_selec, metodo_cod)
            st.session_state['clean_data'] = df
            st.success("Codificación aplicada exitosamente.")
            st.rerun()
            
    # Botón para descargar datos limpios
    st.download_button(
        label="📥 Descargar Dataset Limpio (CSV)",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='dataset_limpio.csv',
        mime='text/csv',
    )
else:
    st.warning("Por favor, carga un dataset en el paso 1 primero.")