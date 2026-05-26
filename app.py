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
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

from components.ui_helpers import setup_branding
setup_branding(
    page_title="Plataforma Integrada de Data Science & ML",
    page_subtitle="Panel ejecutivo para exploración, limpieza y machine learning"
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
    if st.button("Iniciar Pipeline Estándar", use_container_width=True, type="primary"):
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
    if st.button("Iniciar Análisis Avanzado", use_container_width=True, type="primary"):
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

# Información del sistema (estilizado)
sidebar_info_html = '''
<div role="region" aria-label="Información del sistema" style="padding:14px; border-radius:12px; background: linear-gradient(180deg, rgba(2,31,89,0.08), rgba(2,40,115,0.06)); border:1px solid rgba(255,255,255,0.04);">
    <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
        <div>
            <div style="font-size:12px;font-weight:800;color:var(--ui-accent);letter-spacing:0.08em;text-transform:uppercase;">Data Science &amp; ML</div>
            <div style="margin-top:6px;font-size:15px;font-weight:800;color:var(--ui-surface-strong);">Sistema Inicializado Correctamente</div>
        </div>
        <div aria-hidden="true" style="background:rgba(255,255,255,0.06);padding:6px 10px;border-radius:8px;color:var(--ui-surface-strong);font-weight:700;font-size:13px;">v2.0</div>
    </div>
    <hr style="border:none;height:1px;background:rgba(255,255,255,0.03);margin:12px 0;" />
    <p style="margin:0 0 8px 0;color:rgba(245,248,252,0.92);font-size:13px;">Dos formas de usar esta plataforma:</p>
    <ul style="margin:6px 0 12px 18px;color:rgba(245,248,252,0.92);font-size:13px;">
        <li><strong>Pipeline Estándar</strong> — Análisis paso a paso y control total</li>
        <li><strong>Análisis Avanzado</strong> — Visualización interactiva y Deep Learning</li>
    </ul>
    <div style="display:flex;gap:8px;margin-top:6px;">
        <a href="#" style="flex:1;text-align:center;padding:8px 10px;border-radius:8px;background:linear-gradient(135deg,var(--ui-accent),#B96403);color:#FFF;text-decoration:none;font-weight:700;">Guía Rápida</a>
        <a href="#" style="flex:1;text-align:center;padding:8px 10px;border-radius:8px;border:1px solid rgba(255,255,255,0.06);color:var(--ui-surface-strong);text-decoration:none;font-weight:700;">Soporte</a>
    </div>
    <div style="margin-top:10px;font-size:12px;color:rgba(245,248,252,0.7);">Última actualización: Mayo 2026</div>
</div>
'''

st.sidebar.markdown(sidebar_info_html, unsafe_allow_html=True)