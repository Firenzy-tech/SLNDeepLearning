import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
from config.config_loader import Config
from config.processor import DataProcessor
from config.model_service import ModelService
from config.visualizer import Visualizer

def main():
    # 1. Cargar Configuración (antes de set_page_config para obtener el nombre del archivo)
    config = Config()
    data_path = config.get("data_path")

    # --- Determinar nombre dinámico para el título ---
    doc_name = "Weather ML"
    if data_path and os.path.exists(data_path):
        doc_name = os.path.basename(data_path)
    elif "csv_uploader" in st.session_state and st.session_state["csv_uploader"] is not None:
        doc_name = st.session_state["csv_uploader"].name

    st.set_page_config(page_title=f"Clasificador: {doc_name}", layout="wide")
    st.title("Analizador de Datos y Clasificador")

    target_col_default = config.get("target_column")
    model_params = config.get("model_params", {})

    # 2. Carga de Datos (Config o Upload)
    df = None
    if data_path and os.path.exists(data_path):
        st.info(f"Cargando datos desde: {data_path}")
        df = pd.read_csv(data_path)
    else:
        # Instrucciones de uso cuando no hay datos cargados
        st.markdown("""
        ### 📝 Instrucciones para la Carga de Datos
        Para que el analizador funcione correctamente con cualquier dataset, asegúrate de que tu archivo cumpla con lo siguiente:
        1. **Formato**: Debe ser un archivo `.csv` con separador de coma.
        2. **Encabezados**: La primera fila debe contener los nombres de las columnas.
        3. **Columna Objetivo**: Identifica cuál es la variable que quieres predecir (puedes ajustarla en la barra lateral después de subir).
        4. **Limpieza Automática**: No te preocupes por valores faltantes; el sistema aplicará *imputación estadística* (mediana para números, moda para texto) automáticamente.
        """)
        
        uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"], key="csv_uploader")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)

    if df is not None:
        st.subheader("Vista Previa de los Datos (head)")
        st.dataframe(df.head())

        # --- Análisis de Valores Nulos ---
        with st.expander("🔍 Análisis de Valores Nulos (isnull().sum())"):
            st.write(df.isnull().sum())

        # --- Implementación de data.info() ---
        with st.expander("ℹ️ Información Técnica del Dataset (info)"):
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)

        # --- Configuración dinámica del Target ---
        st.sidebar.header("⚙️ Configuración del Modelo")
        all_columns = df.columns.tolist()
        
        
        # Intentar pre-seleccionar la columna del JSON si existe en el CSV
        default_idx = all_columns.index(target_col_default) if target_col_default in all_columns else 0
        
        target_col = st.sidebar.selectbox("🎯 Selecciona la columna objetivo (Target)", options=all_columns, index=default_idx)

        # 2. Selección de Features (Cualquier CSV)
        available_features = [c for c in all_columns if c != target_col]
        selected_features = st.sidebar.multiselect(
            "🛠️ Selecciona las Variables de Entrada (Features)",
            options=available_features,
            default=available_features,
            help="Elimina columnas que sean IDs, nombres o fechas para mejores resultados."
        )

        if not selected_features:
            st.warning("Selecciona al menos una feature para poder entrenar el modelo.")
            return

        # 3. Procesamiento
        processor = DataProcessor()
        try:
            X, y, label_encoder = processor.clean_data(df, target_col, selected_features)
            st.success("Datos procesados correctamente.")

            # Display head of processed data with encoded target for correlation analysis
            st.subheader("Vista Previa de Datos Procesados (para correlación)")
            target_display_name = f"{target_col}_encoded"
            processed_df_for_display = X.copy()
            processed_df_for_display[target_display_name] = y
            st.dataframe(processed_df_for_display.head())

            # --- Análisis de Correlación ---
            with st.expander("📊 Análisis de Correlación"):
                st.write("Analiza la fuerza de la relación lineal entre las variables.")
                
                exclude_target_corr = st.checkbox("Excluir columna objetivo de la matriz", 
                                              help="Útil para detectar multicolinealidad (redundancia) entre tus variables de entrada.")
                
                df_corr = processed_df_for_display.copy()
                if exclude_target_corr:
                    df_corr = df_corr.drop(columns=[target_display_name])
                
                corr_matrix = df_corr.corr()
                
                # Creación del Heatmap optimizado (Tamaño dinámico y fuentes pequeñas)
                n_vars = len(corr_matrix.columns)
                # Calculamos un tamaño que escale con las variables pero con límites razonables
                fig_w = min(10, max(6, n_vars * 0.5))
                fig_h = min(8, max(4, n_vars * 0.4))
                
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                sns.heatmap(
                    corr_matrix, 
                    annot=n_vars < 15,  # Solo muestra números si hay menos de 15 columnas para evitar caos
                    cmap='coolwarm', 
                    fmt=".2f", 
                    ax=ax, 
                    center=0,
                    annot_kws={"size": 7}, # Fuente pequeña para las anotaciones
                    cbar_kws={"shrink": .8}
                )
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.yticks(fontsize=8)
                plt.title("Mapa de Calor de Correlaciones", fontsize=10)
                st.pyplot(fig)

                if not exclude_target_corr:
                    st.write(f"**Correlación con el Objetivo: {target_col}**")
                    # Ordenamos de mayor a menor correlación para identificar las variables más influyentes
                    target_corr = corr_matrix[target_display_name].drop(target_display_name).sort_values(ascending=False)
                    st.bar_chart(target_corr)

            # --- Gráfico de Dispersión Genérico (Personalizado) ---
            with st.expander("🎯 Dispersión Personalizada (Generic Method)"):
                st.write("Genera el gráfico de dispersión exacto definido en el requerimiento.")
                c1, c2 = st.columns(2)
                with c1: gx = st.selectbox("Variable X", X.columns, key="gen_x")
                with c2: gy = st.selectbox("Variable Y", X.columns, key="gen_y", index=min(1, len(X.columns)-1))
                
                fig_gen = Visualizer.plot_generic_scatter(processed_df_for_display, gx, gy, target_display_name)
                st.pyplot(fig_gen)

            # --- Análisis de Relación entre Variables (Scatter & KDE) ---
            with st.expander("📈 Relación entre Variables"):
                st.write("Compara dos variables cualesquiera con respecto al objetivo codificado.")
                
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    feat_x = st.selectbox("Selecciona Eje X", options=X.columns.tolist(), key="feat_x")
                with v_col2:
                    feat_y = st.selectbox("Selecciona Eje Y", options=X.columns.tolist(), 
                                         index=min(1, len(X.columns)-1), key="feat_y")

                fig_rel, (ax_s, ax_k) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Scatter Plot Genérico
                sns.scatterplot(data=processed_df_for_display, x=feat_x, y=feat_y, hue=target_display_name, ax=ax_s, alpha=0.7)
                ax_s.set_title(f'Dispersión: {feat_x} vs {feat_y}')
                
                # KDE Plot Genérico
                sns.kdeplot(data=processed_df_for_display, x=feat_x, y=feat_y, hue=target_display_name, ax=ax_k, fill=True, warn_singular=False)
                ax_k.set_title(f'Densidad (KDE): {feat_x} vs {feat_y}')
                
                st.pyplot(fig_rel)

            # Store processed data in session state for later use if needed
            st.session_state['X_processed'] = X
            st.session_state['y_processed'] = y

        except Exception as e:
            st.error(f"Error procesando datos: {e}")
            return
        # 4. Entrenamiento e Inferencia
        model_service = ModelService(model_params)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Entrenar Modelo"):
                with st.spinner("Entrenando..."):
                    accuracy = model_service.train(st.session_state['X_processed'], st.session_state['y_processed'])
                    st.session_state['model_service'] = model_service
                    st.metric("Precisión del Modelo (Accuracy)", f"{accuracy:.2%}")

        with col2:
            if 'model_service' in st.session_state:
                if st.button("🔮 Realizar Predicciones sobre el set actual"):
                    preds = st.session_state['model_service'].predict(st.session_state['X_processed'])
                    results = pd.DataFrame({"Real": st.session_state['y_processed'], "Predicción": preds})
                    st.write(results.head(10))
    else:
        st.warning("Por favor, proporciona una ruta válida en appsettings.json o sube un archivo.")

if __name__ == "__main__":
    main()