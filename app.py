import streamlit as st

# Configuración global de la página (debe ser la primera llamada a Streamlit)
st.set_page_config(
    page_title="Plataforma de Analítica Avanzada",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicialización del estado global de la sesión
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None
if 'clean_data' not in st.session_state:
    st.session_state['clean_data'] = None

st.title("📊 Plataforma de Data Science & ML")
st.markdown("""
Bienvenido a la plataforma empresarial de análisis de datos. 
Utiliza la barra lateral para navegar a través del pipeline de datos:
1. **Carga y Perfilado**: Importa tu dataset.
2. **Limpieza**: Trata nulos, outliers y codifica variables.
3. **Análisis Visual**: Explora gráficas estadísticas.
4. **Machine/Deep Learning**: Entrena modelos predictivos.
""")

# Información del sistema
st.sidebar.success("Sistema Inicializado. Por favor, ve a 'Carga y Perfilado' para comenzar.")