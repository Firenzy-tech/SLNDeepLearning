import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend para Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import plotly.express as px
import tensorflow as tf
from config import Config, DataProcessor, Visualizer
from config.ann_service import GenericClassifier

# Callback para sincronizar el entrenamiento de Keras con la barra de progreso de Streamlit
class StreamlitProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs, progress_bar, status_text):
        self.epochs = epochs
        self.progress_bar = progress_bar
        self.status_text = status_text

    def on_epoch_end(self, epoch, logs=None):
        percent = (epoch + 1) / self.epochs
        self.progress_bar.progress(percent)
        self.status_text.text(f"Entrenamiento en curso: Época {epoch + 1}/{self.epochs} - Loss: {logs['loss']:.4f}")

def main():
    # 1. Cargar Configuración
    config = Config()
    data_path = config.get("data_path")
    target_col_default = config.get("target_column")
    model_params = config.get("model_params", {})

    # --- Interfaz de Carga de Datos ---
    st.set_page_config(page_title="Analizador de Datos y Clasificador", layout="wide")
    st.title("DeepInsight Analytics Engine")
    
    # Crear columnas para opciones de carga
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Carga de Dataset")
        uploaded_file = st.file_uploader("Seleccione o arrastre un archivo CSV", type=["csv"], key="csv_uploader")
    
    with col2:
        st.subheader("Configuración")
        use_sample = st.checkbox("Usar datos de prueba", value=False, help="Carga el dataset de prueba incluido")

    # 2. Carga de Datos
    df = None
    doc_name = "Analizador ML"
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            doc_name = uploaded_file.name
            st.success(f"✅ Archivo cargado: {doc_name}")
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {e}")
    elif use_sample and data_path and os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            doc_name = os.path.basename(data_path)
            st.info(f"📦 Usando dataset de prueba: {doc_name}")
        except Exception as e:
            st.error(f"❌ Error al cargar datos de prueba: {e}")
    else:
        st.info("""
        **Instrucciones de inicio**
        Cargue un archivo CSV o active el modo de prueba para comenzar el análisis.
        
        **Requisitos del archivo:**
        - Formato: `.csv` con separador de coma
        - Encabezados: Primera fila con nombres de columnas
        - Datos limpios: Sistema aplicará imputación automática
        """)

    if df is not None:
        # Mostrar nombre del archivo/dataset
        st.caption(f"Archivo en análisis: {doc_name}")
        st.divider()
        
        # --- Opciones de Carga en Sidebar ---
        st.sidebar.header("Gestión de Datos")
        st.sidebar.info(f"✅ Archivo cargado: {doc_name}")
        
        col_btn1, col_btn2 = st.sidebar.columns(2)
        with col_btn1:
            if st.button("🔄 Cargar Otro Archivo"):
                st.session_state.clear()
                st.rerun()
        with col_btn2:
            if st.button("🏠 Volver al Inicio"):
                st.session_state.clear()
                st.rerun()
        
        st.sidebar.divider()
        
        # --- Configuración dinámica del Target ---
        st.sidebar.header("Parámetros del Modelo")
        all_columns = df.columns.tolist()

        
        # Intentar pre-seleccionar la columna del JSON si existe en el CSV
        default_idx = all_columns.index(target_col_default) if target_col_default in all_columns else 0
        
        target_col = st.sidebar.selectbox("Variable Objetivo (Target)", options=all_columns, index=default_idx)

        st.sidebar.caption("Este panel entrena un clasificador binario. El objetivo debe tener 2 clases.")

        # 2. Selección de Features (Cualquier CSV)
        available_features = [c for c in all_columns if c != target_col]
        selected_features = st.sidebar.multiselect(
            "Variables de Entrada (Features)",
            options=available_features,
            default=available_features,
            help="Elimina columnas que sean IDs, nombres o fechas para mejores resultados."
        )

        # --- Parámetros de la Red Neuronal ---
        st.sidebar.subheader("🧠 Arquitectura ANN")
        layers_input = st.sidebar.text_input("Capas Ocultas (neuronas)", value="16, 8, 20")
        hidden_layers = [int(x.strip()) for x in layers_input.split(",")]
        dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2)
        l2_val = st.sidebar.select_slider("Regularización L2", options=[0.0001, 0.001, 0.01, 0.1], value=0.001)
        learning_rate = st.sidebar.number_input("Learning Rate", value=0.001, format="%.4f", step=0.0001)
        epochs = st.sidebar.number_input("Épocas", 10, 500, 100)
        batch_size = st.sidebar.number_input("Batch Size", 1, 128, 10)

        if not selected_features:
            st.warning("Selecciona al menos una feature para poder entrenar el modelo.")
            return

        # 3. Procesamiento
        processor = DataProcessor()
        try:
            X, y, label_encoder = processor.clean_data(df, target_col, selected_features)

            unique_target_values = pd.Series(y).dropna().unique()
            if len(unique_target_values) != 2:
                raise ValueError(
                    f"La columna objetivo '{target_col}' tiene {len(unique_target_values)} clases. "
                    "Este panel solo admite clasificación binaria. "
                    "Selecciona una columna categórica/binaria, por ejemplo 'Rain'."
                )

            target_display_name = f"{target_col}_encoded"
            processed_df_for_display = X.copy()
            processed_df_for_display[target_display_name] = y
            
            # Almacenar en session_state antes de las pestañas para que sea global
            st.session_state['X_processed'] = X
            st.session_state['y_processed'] = y

            # --- PANEL DE CONTROL DINÁMICO USANDO TABS ---
            tab_explore, tab_viz, tab_train = st.tabs(["Exploración", "Análisis Visual", "Entrenamiento"])

            with tab_explore:
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Total Registros", df.shape[0])
                kpi2.metric("Features Detectadas", len(selected_features))
                kpi3.metric("Valores Nulos", df.isnull().sum().sum(), delta_color="inverse")
                
                st.divider()
                col_data1, col_data2 = st.columns(2)
                with col_data1:
                    st.markdown("### Previsualización de Datos")
                    st.dataframe(df.head(10), use_container_width=True)
                with col_data2:
                    st.markdown("### Análisis de Valores Nulos")
                    null_df = df.isnull().sum().reset_index()
                    null_df.columns = ['Feature', 'Count']
                    st.plotly_chart(px.bar(null_df, x='Feature', y='Count', color='Count', height=300), use_container_width=True)

                st.subheader("ℹ️ Información Técnica")
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())

                st.subheader("Vista Previa de Datos Procesados")
                st.dataframe(processed_df_for_display.head())

            with tab_viz:
                st.markdown("### Matriz de Correlaciones")
                st.plotly_chart(Visualizer.plot_interactive_corr(processed_df_for_display), use_container_width=True)

                # --- Gráfico de Dispersión Genérico (Personalizado) ---
                st.markdown("### Explorador de Variables")
                c1, c2 = st.columns(2)
                with c1: gx = st.selectbox("Variable X", X.columns, key="gen_x")
                
                # Evitar seleccionar el mismo eje o índice inválido
                gen_y_idx = min(1, len(X.columns)-1) if len(X.columns) > 1 else 0
                available_for_gy = [col for col in X.columns if col != gx]
                if available_for_gy:
                    with c2: gy = st.selectbox("Variable Y", available_for_gy, key="gen_y")
                else:
                    with c2: gy = gx
                
                if gx != gy:
                    fig_px = px.scatter(processed_df_for_display, x=gx, y=gy, color=target_display_name, 
                                      template="plotly_white", marginal_x="box", marginal_y="violin")
                    st.plotly_chart(fig_px, use_container_width=True)

            with tab_train:
                st.markdown("### Centro de Entrenamiento")
                col_train, col_pred = st.columns([2, 1])
                with col_train:
                    if st.button("Iniciar Entrenamiento del Modelo", use_container_width=True, type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        clf = GenericClassifier(df, target_col, selected_features, hidden_layers, dropout, l2_val)
                        clf.preprocess_data()
                        clf.build_model(learning_rate=learning_rate)
                        
                        cb = StreamlitProgressCallback(epochs, progress_bar, status_text)
                        st.session_state['history'] = clf.train(epochs=epochs, batch_size=batch_size, callbacks=[cb])
                        st.session_state['clf'] = clf
                        st.session_state['cm'], st.session_state['report'], st.session_state['y_preds_test'] = clf.evaluate()
                        st.success("Ciclo de entrenamiento finalizado.")

                    # Persistencia de resultados: se muestran si existen en el estado de la sesión
                    if 'clf' in st.session_state and 'history' in st.session_state:
                        clf_trained = st.session_state['clf']
                        st.plotly_chart(Visualizer.plot_interactive_loss(st.session_state['history']), use_container_width=True)
                        
                        m1, m2 = st.columns(2)
                        m1.metric("Test Accuracy", f"{st.session_state['report']['accuracy']:.2%}")
                        m2.metric("F1-Score", f"{st.session_state['report']['macro avg']['f1-score']:.2f}")
                        
                        st.write("### Evaluación de Clasificación")
                        y_preds_test = st.session_state['y_preds_test']
                        if y_preds_test.ndim > 1:
                            y_preds_test = y_preds_test.flatten()
                        fig_cm = Visualizer.plot_confusion_matrix(clf_trained.y_test, y_preds_test)
                        st.pyplot(fig_cm)
                        plt.close(fig_cm)

                        if st.button("Exportar Modelo Entrenado"):
                            clf_trained.save_assets("ann_classifier_output")
                            st.success("Activos exportados al directorio local.")

                with col_pred:
                    st.subheader("Inferencia Predictiva")
                    if 'clf' in st.session_state:
                        if st.button("Procesar Predicciones (Test Set)"):
                            clf = st.session_state['clf']
                            y_prob = clf.model.predict(clf.X_test)
                            preds = (y_prob > 0.5).astype(int).flatten()
                            results = pd.DataFrame({
                                "Real": clf.y_test, 
                                "Predicción": preds,
                                "Probabilidad (%)": (y_prob.flatten() * 100).round(2)
                            })
                            st.write("Comparación de predicciones:")
                            st.dataframe(results.head(15))
                    else:
                        st.warning("Primero debes completar un ciclo de entrenamiento.")

        except Exception as e:
            st.error(f"Error en el flujo del panel: {e}")
            return
    else:
        st.warning("Por favor, proporciona una ruta válida en appsettings.json o sube un archivo.")

if __name__ == "__main__":
    main()