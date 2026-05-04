import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend para Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
from config import Config, DataProcessor, Visualizer
from config.ann_service import GenericClassifier

def main():
    # 1. Cargar Configuración
    config = Config()
    data_path = config.get("data_path")
    target_col_default = config.get("target_column")
    model_params = config.get("model_params", {})

    # --- Interfaz de Carga de Datos ---
    st.set_page_config(page_title="Analizador de Datos y Clasificador", layout="wide")
    st.title("📊 Analizador de Datos y Clasificador")
    
    # Crear columnas para opciones de carga
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Cargar Datos")
        uploaded_file = st.file_uploader("Selecciona tu archivo CSV", type=["csv"], key="csv_uploader")
    
    with col2:
        st.subheader("📦 Datos Disponibles")
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
        👋 **Bienvenido al Analizador de Datos y Clasificador**
        
        1. 📁 Carga tu archivo CSV usando el botón arriba
        2. 📦 O activa "Usar datos de prueba" para probar con un dataset incluido
        
        **Requisitos del archivo:**
        - Formato: `.csv` con separador de coma
        - Encabezados: Primera fila con nombres de columnas
        - Datos limpios: Sistema aplicará imputación automática
        """)

    if df is not None:
        # Mostrar nombre del archivo/dataset
        st.markdown(f"### 📄 Analizando: **{doc_name}**")
        st.divider()
        
        # --- Opciones de Carga en Sidebar ---
        st.sidebar.header("📂 Gestión de Datos")
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
        st.sidebar.header("⚙️ Configuración del Modelo")
        all_columns = df.columns.tolist()

        
        # Intentar pre-seleccionar la columna del JSON si existe en el CSV
        default_idx = all_columns.index(target_col_default) if target_col_default in all_columns else 0
        
        target_col = st.sidebar.selectbox("🎯 Selecciona la columna objetivo (Target)", options=all_columns, index=default_idx)

        st.sidebar.caption("Este panel entrena un clasificador binario. El objetivo debe tener 2 clases.")

        # 2. Selección de Features (Cualquier CSV)
        available_features = [c for c in all_columns if c != target_col]
        selected_features = st.sidebar.multiselect(
            "🛠️ Selecciona las Variables de Entrada (Features)",
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
            tab_explore, tab_viz, tab_train = st.tabs(["📋 Exploración", "📊 Visualización", "🧠 Entrenamiento y Ajuste"])

            with tab_explore:
                st.subheader("Vista Previa de Datos (Original)")
                st.dataframe(df.head())

                st.subheader("🔍 Análisis de Valores Nulos")
                st.write(df.isnull().sum())

                st.subheader("ℹ️ Información Técnica")
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())

                st.subheader("Vista Previa de Datos Procesados")
                st.dataframe(processed_df_for_display.head())

            with tab_viz:
                st.subheader("📊 Análisis de Correlación")
                st.write("Analiza la fuerza de la relación lineal entre las variables.")
                
                exclude_target_corr = st.checkbox("Excluir columna objetivo de la matriz", 
                                              help="Útil para detectar multicolinealidad (redundancia) entre tus variables de entrada.")
                
                df_corr = processed_df_for_display.copy()
                if exclude_target_corr:
                    df_corr = df_corr.drop(columns=[target_display_name])
                
                corr_matrix = df_corr.corr()
                
                if corr_matrix.empty or corr_matrix.dropna(how='all').dropna(axis=1, how='all').empty:
                    st.warning("⚠️ No hay suficientes variables numéricas o varianza en los datos para generar una matriz de correlación.")
                else:
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
                    plt.close(fig)

                if not exclude_target_corr:
                    st.write(f"**Correlación con el Objetivo: {target_col}**")
                    # Ordenamos de mayor a menor correlación para identificar las variables más influyentes
                    target_corr = corr_matrix[target_display_name].drop(target_display_name).sort_values(ascending=False)
                    fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
                    target_corr.plot(kind='barh', ax=ax_corr, color='steelblue')
                    ax_corr.set_xlabel('Correlación')
                    ax_corr.set_title(f'Correlación con {target_col}')
                    plt.tight_layout()
                    st.pyplot(fig_corr)
                    plt.close(fig_corr)

                # --- Análisis de Outliers y Distribución ---
                st.divider()
                st.write("Visualiza la dispersión y los valores atípicos de tus variables numéricas.")
                num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                if num_cols:
                    dist_col = st.selectbox("Selecciona variable para analizar:", num_cols)
                    fig_dist = Visualizer.plot_distribution(X, dist_col)
                    st.pyplot(fig_dist)
                    plt.close(fig_dist)
                else:
                    st.info("No hay variables numéricas para mostrar distribuciones.")

                # --- Gráfico de Dispersión Genérico (Personalizado) ---
                st.divider()
                st.write("Genera el gráfico de dispersión exacto definido en el requerimiento.")
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
                    fig_gen = Visualizer.plot_generic_scatter(processed_df_for_display, gx, gy, target_display_name)
                    st.pyplot(fig_gen)
                    plt.close(fig_gen)
                else:
                    st.warning("Selecciona dos variables diferentes para el gráfico.")

                # --- Análisis de Relación entre Variables (Scatter & KDE) ---
                st.divider()
                st.write("Compara dos variables cualesquiera con respecto al objetivo codificado.")
                
                if len(X.columns) < 2:
                    st.warning("Se necesitan al menos 2 variables para hacer este análisis.")
                else:
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        feat_x = st.selectbox("Selecciona Eje X", options=X.columns.tolist(), key="feat_x")
                    with v_col2:
                        available_for_fy = [col for col in X.columns.tolist() if col != feat_x]
                        if available_for_fy:
                            feat_y = st.selectbox("Selecciona Eje Y", options=available_for_fy, key="feat_y")
                        else:
                            feat_y = feat_x

                    try:
                        fig_rel, (ax_s, ax_k) = plt.subplots(1, 2, figsize=(14, 5))
                        
                        # Scatter Plot Genérico
                        sns.scatterplot(data=processed_df_for_display, x=feat_x, y=feat_y, hue=target_display_name, ax=ax_s, alpha=0.7)
                        ax_s.set_title(f'Dispersión: {feat_x} vs {feat_y}')
                        
                        # KDE Plot Genérico - Solo si hay suficientes datos
                        try:
                            sns.kdeplot(data=processed_df_for_display, x=feat_x, y=feat_y, hue=target_display_name, ax=ax_k, fill=True, warn_singular=False)
                        except Exception as kde_err:
                            ax_k.text(0.5, 0.5, f'No se puede graficar KDE: {str(kde_err)[:30]}', 
                                     ha='center', va='center', transform=ax_k.transAxes)
                        ax_k.set_title(f'Densidad (KDE): {feat_x} vs {feat_y}')
                        
                        st.pyplot(fig_rel)
                        plt.close(fig_rel)
                    except Exception as plot_err:
                        st.error(f"Error al crear el gráfico: {plot_err}")
                plt.close(fig_rel)

            with tab_train:
                st.header("🛠️ Tablero de Entrenamiento")
                st.info("Configura la arquitectura en la barra lateral. Al presionar 'Entrenar', se aplicarán los ajustes actuales.")
                
                col_train, col_pred = st.columns(2)
                with col_train:
                    if st.button("🚀 Iniciar Entrenamiento"):
                        with st.spinner("Optimizando Red Neuronal..."):
                            # Instancia del clasificador con parámetros de la interfaz
                            clf = GenericClassifier(df, target_col, selected_features, hidden_layers, dropout, l2_val)
                            clf.preprocess_data()
                            clf.build_model(learning_rate=learning_rate)
                            st.session_state['history'] = clf.train(epochs=epochs, batch_size=batch_size)
                            st.session_state['clf'] = clf
                            st.session_state['cm'], st.session_state['report'], st.session_state['y_preds_test'] = clf.evaluate()

                    # Persistencia de resultados: se muestran si existen en el estado de la sesión
                    if 'clf' in st.session_state and 'history' in st.session_state:
                        clf_trained = st.session_state['clf']
                        st.subheader("📈 Rendimiento del Entrenamiento")
                        fig_history = Visualizer.plot_training_history(st.session_state['history'])
                        st.pyplot(fig_history)
                        plt.close(fig_history)

                        st.metric("Accuracy Final", f"{st.session_state['report']['accuracy']:.2%}")
                        st.write("### 📝 Reporte Detallado")
                        st.dataframe(pd.DataFrame(st.session_state['report']).transpose())
                        
                        st.write("### 📉 Matriz de Confusión")
                        y_preds_test = st.session_state['y_preds_test']
                        if y_preds_test.ndim > 1:
                            y_preds_test = y_preds_test.flatten()
                        fig_cm = Visualizer.plot_confusion_matrix(clf_trained.y_test, y_preds_test)
                        st.pyplot(fig_cm)
                        plt.close(fig_cm)

                        if st.button("💾 Guardar Activos del Modelo"):
                            clf_trained.save_assets("ann_classifier_output")
                            st.success("Modelo y activos guardados correctamente.")

                with col_pred:
                    st.subheader("🔮 Inferencia en Tiempo Real")
                    if 'clf' in st.session_state:
                        if st.button("🧪 Ejecutar Predicciones sobre Test"):
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