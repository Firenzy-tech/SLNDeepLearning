"""Análisis Avanzado: Aplicación de Clasificación ANN con Visualización Interactiva

Este módulo combina:
- Carga y preprocesamiento de datos
- Visualización interactiva con Plotly
- Entrenamiento de clasificadores ANN
- Evaluación y exportación de modelos
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import zipfile
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend para Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import io
import plotly.express as px
import tensorflow as tf
from config import Config, DataProcessor, Visualizer, GenericClassifier
from utils.groq_diagnostic import GroqDiagnostician


class StreamlitProgressCallback(tf.keras.callbacks.Callback):
    """Actualiza la barra de progreso de Streamlit al final de cada época."""

    def __init__(self, epochs, progress_bar, status_text):
        self.epochs = epochs
        self.progress_bar = progress_bar
        self.status_text = status_text

    def on_epoch_end(self, epoch, logs=None):
        percent = (epoch + 1) / self.epochs
        self.progress_bar.progress(percent)
        self.status_text.text(f"Entrenamiento en curso: Época {epoch + 1}/{self.epochs} - Loss: {logs['loss']:.4f}")


def main():
    """Construye la interfaz avanzada de análisis y entrenamiento ANN."""

    # Configuración de la página
    st.set_page_config(page_title="Análisis Avanzado", layout="wide", page_icon="🧠")
    
    from components.ui_helpers import setup_branding, show_spinner
    setup_branding()
    
    st.header("Análisis Avanzado - Clasificador ANN")
    st.markdown("""
    Plataforma completa para análisis exploratorio, visualización interactiva y entrenamiento 
    de redes neuronales artificiales (ANN).
    """)
    
    # 1) Cargar configuración base
    config = Config()
    data_path = config.get("data_path")
    target_col_default = config.get("target_column")
    model_params = config.get("model_params", {})

    # 2) Carga de archivo
    st.divider()
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Cargar Dataset")
        uploaded_file = st.file_uploader("Seleccione o arrastre un archivo CSV", type=["csv"], key="csv_uploader")
    
    with col2:
        use_sample = st.checkbox("Usar datos de ejemplo", value=False)

    # 3) Carga del dataset
    loaded_dataset = None
    doc_name = "Análisis Avanzado"
    
    if uploaded_file is not None:
        try:
            loaded_dataset = pd.read_csv(uploaded_file)
            doc_name = uploaded_file.name
            st.success(f"Archivo cargado: {doc_name}")
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {e}")
    elif use_sample and data_path and os.path.exists(data_path):
        try:
            loaded_dataset = pd.read_csv(data_path)
            doc_name = os.path.basename(data_path)
            st.info(f"Usando dataset de prueba: {doc_name}")
        except Exception as e:
            st.error(f"❌ Error al cargar datos de prueba: {e}")
    else:
        st.info("""
        **Instrucciones:**
        1. Cargue un archivo CSV o active el modo de prueba
        2. Configure los parámetros en la barra lateral
        3. Explore los datos y entrene el modelo
        
        **Requisitos del archivo:**
        - Formato: `.csv` con separador de coma
        - Encabezados: Primera fila con nombres de columnas
        - Datos semi-limpios: Sistema aplicará imputación automática
        """)

    if loaded_dataset is not None:
        st.caption(f"Análisis de: **{doc_name}** ({loaded_dataset.shape[0]} filas × {loaded_dataset.shape[1]} columnas)")
        st.divider()
        
        # 4) Sidebar: Configuración
        st.sidebar.header("⚙️ Configuración")
        st.sidebar.info(f"Archivo: {doc_name}")
        
        # Botones de control
        col_btn1, col_btn2 = st.sidebar.columns(2)
        with col_btn1:
            if st.button("🔄 Cargar Otro"):
                st.session_state.clear()
                st.rerun()
        with col_btn2:
            if st.button("🏠 Ir al Inicio"):
                st.session_state.clear()
                st.rerun()
        
        st.sidebar.divider()
        
        # Selección de columnas
        st.sidebar.header("📊 Parámetros del Análisis")
        all_columns = loaded_dataset.columns.tolist()
        default_idx = all_columns.index(target_col_default) if target_col_default in all_columns else 0
        
        target_col = st.sidebar.selectbox("Variable Objetivo (Target)", options=all_columns, index=default_idx)
        st.sidebar.caption("⚠️ El objetivo debe tener exactamente 2 clases para clasificación binaria.")

        available_features = [c for c in all_columns if c != target_col]
        selected_features = st.sidebar.multiselect(
            "Variables de Entrada (Features)",
            options=available_features,
            default=available_features,
            help="Elimina IDs, nombres o fechas para mejores resultados."
        )

        # Parámetros ANN
        st.sidebar.subheader("Arquitectura ANN")
        layers_input = st.sidebar.text_input("Capas Ocultas (neuronas)", value="16, 8, 20")
        try:
            hidden_layers = [int(x.strip()) for x in layers_input.split(",")]
        except ValueError:
            st.sidebar.error("Formato inválido. Use: 16, 8, 20")
            hidden_layers = [16, 8, 20]
        
        dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2)
        l2_val = st.sidebar.select_slider("Regularización L2", options=[0.0001, 0.001, 0.01, 0.1], value=0.001)
        learning_rate = st.sidebar.number_input("Learning Rate", value=0.001, format="%.4f", step=0.0001)
        epochs = st.sidebar.number_input("Épocas", 10, 500, 100)
        batch_size = st.sidebar.number_input("Batch Size", 1, 128, 10)

        if not selected_features:
            st.warning(" Selecciona al menos una feature para continuar.")
            return

        # 5) Preprocesamiento
        processor = DataProcessor()
        try:
            X, y, label_encoder = processor.clean_data(loaded_dataset, target_col, selected_features)

            unique_target_values = pd.Series(y).dropna().unique()
            if len(unique_target_values) != 2:
                st.error(
                    f"❌ La columna '{target_col}' tiene {len(unique_target_values)} clases. "
                    "Este panel solo admite clasificación binaria (2 clases)."
                )
                return

            # Preparar dataframes para visualización
            target_display_name = f"{target_col} (Target)"
            processed_df_for_display = X.copy()
            if label_encoder:
                processed_df_for_display[target_display_name] = label_encoder.inverse_transform(y)
            else:
                processed_df_for_display[target_display_name] = y.astype(str)
            
            raw_viz_df = loaded_dataset.copy()
            raw_viz_df[target_display_name] = processed_df_for_display[target_display_name]

            st.session_state['processed_viz_df'] = processed_df_for_display
            st.session_state['raw_viz_df'] = raw_viz_df

            # 6) Interfaz con tabs
            tab_explore, tab_viz, tab_train = st.tabs(["📊 Exploración", "📈 Visualización", "🚀 Entrenamiento"])

            # TAB 1: EXPLORACIÓN
            with tab_explore:
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric(" Registros", f"{loaded_dataset.shape[0]:,}")
                kpi2.metric(" Features", len(selected_features))
                kpi3.metric(" Valores Nulos", int(loaded_dataset.isnull().sum().sum()), delta_color="inverse")
                kpi4.metric(" Clases", len(unique_target_values))
                
                st.divider()
                
                col_data1, col_data2 = st.columns(2)
                
                with col_data1:
                    st.markdown("### Previsualización de Datos")
                    st.dataframe(loaded_dataset.head(10), use_container_width=True)
                
                with col_data2:
                    st.markdown("### Calidad de Datos")
                    null_df = loaded_dataset.isnull().sum().reset_index()
                    null_df.columns = ['Feature', 'Nulos']
                    null_df = null_df[null_df['Nulos'] > 0].sort_values('Nulos', ascending=False)
                    if len(null_df) > 0:
                        fig_null = px.bar(null_df, x='Feature', y='Nulos', color='Nulos', height=300)
                        st.plotly_chart(fig_null, use_container_width=True)
                    else:
                        st.success(" Sin valores nulos detectados")

                st.subheader(" Estadísticas Descriptivas")
                df_numeric = loaded_dataset.select_dtypes(include=[np.number])
                if not df_numeric.empty:
                    st.dataframe(df_numeric.describe().T, use_container_width=True)
                else:
                    st.info("No se detectaron columnas numéricas.")

            # TAB 2: VISUALIZACIÓN
            with tab_viz:
                processed_viz_df = st.session_state['processed_viz_df']
                raw_viz_df = st.session_state['raw_viz_df']
                
                with st.expander("🔗 Matriz de Correlaciones Interactiva", expanded=True):
                    try:
                        fig_corr = Visualizer.plot_interactive_corr(processed_viz_df)
                        st.plotly_chart(fig_corr, use_container_width=True)
                    except Exception as e:
                        st.warning(f"No se pudo generar correlaciones: {e}")

                st.divider()
                st.subheader("🔍 Análisis Detallado por Variable")
                
                col_sel1, col_sel2 = st.columns([2, 1])
                with col_sel1:
                    selected_col = st.selectbox("Seleccione una variable:", options=selected_features)
                
                if selected_col:
                    is_numeric = np.issubdtype(loaded_dataset[selected_col].dtype, np.number)
                    
                    v_col1, v_col2 = st.columns([2, 1])
                    
                    with v_col1:
                        if is_numeric:
                            fig_hist = px.histogram(
                                raw_viz_df, x=selected_col, color=target_display_name,
                                marginal="box", barmode="overlay",
                                title=f"Distribución: {selected_col}",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        else:
                            fig_bar = px.histogram(
                                raw_viz_df, x=selected_col, color=target_display_name,
                                barmode="group", title=f"Frecuencia: {selected_col}",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)
                    
                    with v_col2:
                        if is_numeric:
                            fig_box = px.box(
                                raw_viz_df, x=target_display_name, y=selected_col, 
                                color=target_display_name, title="Separación de Clases",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig_box, use_container_width=True)

                st.divider()
                
                with st.expander("🔗 Explorador de Relaciones Bivariadas"):
                    c1, c2 = st.columns(2)
                    with c1: 
                        gx = st.selectbox("Eje X", selected_features, key="viz_x")
                    with c2: 
                        gy = st.selectbox("Eje Y", [c for c in selected_features if c != gx], key="viz_y")
                    
                    fig_rel = px.scatter(
                        raw_viz_df, x=gx, y=gy, color=target_display_name,
                        trendline="ols" if (np.issubdtype(raw_viz_df[gx].dtype, np.number) and 
                                           np.issubdtype(raw_viz_df[gy].dtype, np.number)) else None,
                        title=f"{gx} vs {gy}",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_rel, use_container_width=True)

            # TAB 3: ENTRENAMIENTO
            with tab_train:
                st.markdown("###  Centro de Entrenamiento")
                col_train, col_pred = st.columns([2, 1])
                
                with col_train:
                    if st.button(" Iniciar Entrenamiento", use_container_width=True, type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            clf = GenericClassifier(loaded_dataset, target_col, selected_features, 
                                                  hidden_layers, dropout, l2_val)
                            clf.preprocess_data()
                            clf.build_model(learning_rate=learning_rate)
                            
                            cb = StreamlitProgressCallback(epochs, progress_bar, status_text)
                            st.session_state['history'] = clf.train(epochs=epochs, batch_size=batch_size, callbacks=[cb])
                            st.session_state['clf'] = clf
                            st.session_state['cm'], st.session_state['report'], st.session_state['y_preds_test'] = clf.evaluate()
                            st.success(" Entrenamiento completado exitosamente.")
                        except Exception as e:
                            st.error(f"❌ Error en entrenamiento: {e}")

                    # Mostrar resultados si existen
                    if 'clf' in st.session_state and 'history' in st.session_state:
                        clf_trained = st.session_state['clf']
                        
                        try:
                            st.plotly_chart(Visualizer.plot_interactive_loss(st.session_state['history']), 
                                          use_container_width=True)
                        except Exception as e:
                            st.warning(f"No se pudo generar gráfico de pérdida: {e}")
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric(" Accuracy", f"{st.session_state['report']['accuracy']:.2%}")
                        m2.metric(" F1-Score", f"{st.session_state['report']['macro avg']['f1-score']:.2f}")
                        m3.metric(" Precision", f"{st.session_state['report']['weighted avg']['precision']:.2f}")
                        
                        st.write("### Matriz de Confusión")
                        y_preds_test = st.session_state['y_preds_test']
                        if y_preds_test.ndim > 1:
                            y_preds_test = y_preds_test.flatten()
                        
                        try:
                            fig_cm = Visualizer.plot_confusion_matrix(clf_trained.y_test, y_preds_test)
                            st.pyplot(fig_cm)
                            plt.close(fig_cm)
                        except Exception as e:
                            st.warning(f"No se pudo generar matriz de confusión: {e}")

                        st.divider()
                        
                        # 🆕 SECCIÓN: Diagnóstico del Modelo (Lenguaje No Técnico)
                        with st.expander("🔍 Diagnóstico del Modelo (Explicación Clara)", expanded=False):
                            st.markdown("""
                            **¿Cómo funciona mi modelo?** Dejaremos que una IA te explique los resultados 
                            de forma fácil de entender, sin tecnicismos.
                            """)
                            
                            if 'report' not in st.session_state:
                                st.warning("⚠️ Primero entrena el modelo para generar métricas. Haz clic en 'Iniciar Entrenamiento'.")
                            else:
                                if st.button("📊 Generar Diagnóstico", use_container_width=True):
                                    with show_spinner("Generando explicabilidad (SHAP)..."):
                                        try:
                                            import shap
                                            clf_trained = st.session_state['clf']
                                            shap_values, X_sample, expected_value = clf_trained.explain_model(sample_size=50)
                                            
                                            feature_names = clf_trained.processed_features
                                            mean_abs_shap = np.abs(shap_values).mean(axis=0)
                                            if mean_abs_shap.ndim > 1:
                                                mean_abs_shap = mean_abs_shap.flatten()
                                                
                                            shap_dict = dict(zip(feature_names, mean_abs_shap))
                                            shap_sorted = sorted(shap_dict.items(), key=lambda item: item[1], reverse=True)
                                            top_features_str = ", ".join([f"{k} (impacto: {v:.4f})" for k, v in shap_sorted[:5]])
                                            
                                            st.session_state['shap_values'] = shap_values
                                            st.session_state['shap_X_sample'] = X_sample
                                            st.session_state['shap_features'] = feature_names
                                            st.session_state['shap_summary'] = top_features_str
                                        except Exception as e:
                                            st.warning(f"No se pudo generar explicabilidad SHAP: {e}")
                                            st.session_state['shap_summary'] = None

                                    with show_spinner("Analizando resultados con Groq IA..."):
                                        try:
                                            # Extrae métricas del reporte
                                            report = st.session_state['report']
                                            metrics = {
                                                'accuracy': report.get('accuracy', 0),
                                                'precision': report.get('weighted avg', {}).get('precision', 0),
                                                'recall': report.get('weighted avg', {}).get('recall', 0),
                                                'f1_score': report.get('weighted avg', {}).get('f1-score', 0),
                                            }

                                            # Inicializa Groq y genera diagnóstico
                                            diagnostician = GroqDiagnostician()
                                            diagnostic = diagnostician.generate_diagnostic(
                                                metrics=metrics,
                                                model_name=f"Clasificador ANN - {doc_name}",
                                                shap_summary=st.session_state.get('shap_summary')
                                            )

                                            st.session_state['diagnostic'] = diagnostic

                                        except ValueError as e:
                                            st.warning(f"⚙️ {str(e)}")
                                            st.info(
                                                """
                                                **Para usar esta función:**
                                                1. Crea una API Key en [Groq Console](https://console.groq.com)
                                                2. En tu terminal, ejecuta:
                                                   ```bash
                                                   setx GROQ_API_KEY "tu_token_aqui"
                                                   ```
                                                3. Reinicia Streamlit: `streamlit run app.py`
                                                """
                                            )
                                        except Exception as e:
                                            st.error(f"❌ Error al generar diagnóstico: {str(e)}")
                            
                        # Mostrar SHAP y diagnóstico (fuera del expander para garantizar visibilidad)
                        if 'shap_values' in st.session_state:
                            st.markdown("---")
                            st.markdown("### 🎯 Impacto de las Variables (SHAP)")
                            st.markdown("Este gráfico muestra cómo cada variable influye en las predicciones. Los valores más a la derecha tienen un impacto positivo fuerte en el resultado.")
                            try:
                                import shap
                                plt.clf()
                                shap.summary_plot(
                                    st.session_state['shap_values'], 
                                    st.session_state['shap_X_sample'], 
                                    feature_names=st.session_state['shap_features'], 
                                    show=False
                                )
                                st.pyplot(plt.gcf())
                                plt.clf()
                            except Exception as e:
                                st.warning(f"Error al renderizar el gráfico SHAP: {e}")

                        if 'diagnostic' in st.session_state:
                            st.markdown("---")
                            st.markdown("### 📋 Diagnóstico (Explicación No Técnica):")
                            st.info(st.session_state['diagnostic'])
                        
                        st.divider()
                        
                        if st.button(" Preparar Descarga del Modelo", use_container_width=True):
                            try:
                                with tempfile.TemporaryDirectory() as tmpdir:
                                    base_path = os.path.join(tmpdir, "ann_classifier_output")
                                    clf_trained.save_assets(base_path)
                                    
                                    zip_buffer = io.BytesIO()
                                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                                        for root, _, files in os.walk(tmpdir):
                                            for file in files:
                                                zf.write(os.path.join(root, file), arcname=file)
                                    
                                    st.session_state['download_zip'] = zip_buffer.getvalue()
                                    st.success("✅ Modelo empaquetado.")
                            except Exception as e:
                                st.error(f"Error al preparar descarga: {e}")

                        if 'download_zip' in st.session_state:
                            st.download_button(
                                label="📥 Descargar Modelo (.zip)",
                                data=st.session_state['download_zip'],
                                file_name="modelo_entrenado.zip",
                                mime="application/zip",
                                use_container_width=True
                            )

                with col_pred:
                    st.subheader("🔮 Inferencia")
                    if 'clf' in st.session_state:
                        if st.button("Procesar Predicciones"):
                            try:
                                clf = st.session_state['clf']
                                y_prob = clf.model.predict(clf.X_test)
                                preds = (y_prob > 0.5).astype(int).flatten()
                                results = pd.DataFrame({
                                    "Real": clf.y_test, 
                                    "Predicción": preds,
                                    "Probabilidad (%)": (y_prob.flatten() * 100).round(2)
                                })
                                st.dataframe(results.head(20), use_container_width=True)
                            except Exception as e:
                                st.error(f"Error en inferencia: {e}")
                    else:
                        st.warning("⚠️ Completa un entrenamiento primero.")

        except Exception as e:
            st.error(f"❌ Error en el flujo: {str(e)}")
            st.info("Verifica que todas las dependencias estén instaladas: `pip install -r requirements.txt`")

    else:
        st.warning("📂 Proporciona un archivo CSV o activa el modo de prueba.")


if __name__ == "__main__":
    main()
