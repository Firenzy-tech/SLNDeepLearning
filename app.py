import streamlit as st
import os
from pathlib import Path

# Carga variables de entorno desde .env (para API Key de Groq, etc.)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # Si dotenv no está instalado, continúa sin él

# Configuración global de la página (debe ser la primera llamada a Streamlit)
st.set_page_config(
    page_title="Plataforma de Data Science & ML",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicialización del estado global de la sesión
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None
if 'clean_data' not in st.session_state:
    st.session_state['clean_data'] = None

# Página de inicio
st.title("Plataforma Integrada de Data Science & ML")

st.markdown("""
Bienvenido a la plataforma empresarial de análisis de datos y machine learning. 
Tenemos dos flujos de trabajo disponibles optimizados para diferentes necesidades:
""")

st.divider()

# Mostrar las dos opciones disponibles
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Proceso Base
    Análisis paso a paso con control total sobre cada etapa:
    - **Carga y Perfilado**: Importa y explora tu dataset
    - **Limpieza**: Maneja nulos, outliers y codificación
    - **Análisis Visual**: Gráficas estadísticas detalladas
    - **Machine Learning**: Entrena modelos tradicionales
    
    **Ideal para:** Análisis exploratorio, limpieza manual y modelos clásicos.
    """)
    if st.button("Iniciar Pipeline Estándar", use_container_width=True, type="secondary"):
        st.switch_page("pages/1_Carga_y_Perfilado.py")

with col2:
    st.markdown("""
    ### Análisis Avanzado
    Plataforma completa con visualización interactiva y Deep Learning:
    -  Carga rápida de datos
    -  Visualizaciones con Plotly
    -  Entrenamiento de redes neuronales (ANN)
    -  Predicciones en tiempo real
    
    **Ideal para:** Análisis rápido, Deep Learning y visualización avanzada.
    """)
    if st.button("Iniciar Análisis Avanzado", use_container_width=True, type="secondary"):
        st.switch_page("pages/5_Advanced_Analysis.py")

st.divider()

# Información de ayuda
st.markdown("""
## Guía Rápida

### Pipeline Estándar (izquierda)
Sigue estos pasos en orden:
1. Ve a **Carga y Perfilado** y sube tu dataset
2. Ve a **Limpieza de Datos** para tratamiento de nulos
3. Usa **Análisis y Gráficas** para exploración visual
4. Entrena modelos en **Machine Learning**

### Análisis Avanzado (derecha)
Todo en una sola interfaz:
- Carga datos directamente
- Visualiza correlaciones e interacciones
- Entrena redes neuronales con arquitectura personalizable
- Descarga los modelos entrenados

## Requisitos
- Archivos CSV con encabezados
- Variables objetivo con 2 clases (clasificación binaria)
- Dependencias: `pip install -r requirements.txt`

## Instalación de Dependencias
```bash
pip install -r requirements.txt
```

## Soporte
Si encuentras errores de importación, asegúrate de:
1. Tener todas las dependencias instaladas
2. Estar en la carpeta raíz del proyecto
3. Ejecutar con: `streamlit run app.py`
""")

# Información del sistema
st.sidebar.success("Sistema Inicializado Correctamente")
st.sidebar.divider()
st.sidebar.info("""
**Dos formas de usar esta plataforma:**

**Pipeline Estándar**
- Análisis paso a paso
- Control completo
- Ideal para exploración

**Análisis Avanzado**
- Todo integrado
- Visualización interactiva
- Ideal para Deep Learning
""")

st.sidebar.divider()
st.sidebar.markdown("""
**Versión:** 2.0 
**Última actualización:** Mayo 2026
""")