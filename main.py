import streamlit as st
from config.config_loader import Config

@st.cache_resource
def get_config():
    return Config()

def main():
    # Configuración de la página de Streamlit
    # Verificamos si estamos dentro del contexto de Streamlit
    if not st.runtime.exists():
        print("\n" + "!"*50)
        print("ERROR: Contexto de Streamlit no detectado.")
        print("Por favor, ejecuta el programa usando: streamlit run main.py")
        print("!"*50 + "\n")
        return

    st.set_page_config(page_title="Deep Learning Control Panel", layout="wide")

    # Obtener configuración (cacheadat)
    config = get_config()
    
    st.title("🧠 Panel de Control de Deep Learning")
    st.sidebar.header("Menú de Configuración")

    model_config = config.get("model", {})
    hyperparameters = model_config.get("hyperparameters", {})

    # Widgets interactivos en la barra lateral basados en appsettings.json
    lr = st.sidebar.number_input("Learning Rate", value=hyperparameters.get("learning_rate", 0.001), format="%.4f")
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    epochs = st.sidebar.slider("Epochs", 1, 100, hyperparameters.get("epochs", 50))

    # Área principal para estadísticas y estado
    st.subheader("Estado del Sistema e Hiperparámetros")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Modelo:** {model_config.get('type')}")
        st.info(f"**Dataset:** {config.get('dataset', {}).get('path')}")
    
    if st.button("🚀 Iniciar Entrenamiento"):
        st.success(f"Entrenamiento iniciado con LR={lr}, Batch={batch_size}, Epochs={epochs}")

if __name__ == "__main__":
    main()