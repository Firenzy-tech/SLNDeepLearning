# SLNDeepLearning

Este proyecto proporciona una infraestructura base para experimentos de Deep Learning y Machine Learning, integrando un sistema de configuración centralizado, una interfaz gráfica interactiva y un clasificador neuronal configurable.

## Documentación Rápida

Si quieres entender el flujo sin leer primero el código, empieza por [GUIA_ARQUITECTURA.md](GUIA_ARQUITECTURA.md). Ahí se resume la entrada principal, el preprocesamiento, el entrenamiento y la exportación del modelo.

## Requisitos Previos

- Python 3.8 o superior.
- Pip (gestor de paquetes de Python).

## Instalación y Configuración

Para mantener un entorno limpio y evitar conflictos de librerías, se recomienda el uso de un entorno virtual:

1. **Clonar o acceder al directorio del proyecto:**
   ```bash
   cd c:\Users\Juan\source\repos\SLNDeepLearning
   ```

2. **Crear un entorno virtual:**
   ```bash
   python -m venv venv
   ```

3. **Activar el entorno virtual:**
   - **Windows:**
     ```bash
     .\venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Instalar las dependencias necesarias:**
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución

El proyecto utiliza **Streamlit** para la interfaz gráfica. Para iniciar el panel de control, ejecuta desde la raíz del proyecto:

```bash
streamlit run app.py
```

El navegador se abrirá automáticamente en `http://localhost:8501` donde podrás cargar datos, explorarlos, entrenar el modelo y descargar los artefactos.

## Configuración de Hiperparámetros

Puedes modificar los valores por defecto editando el archivo `config/appsettings.json`. El cargador de configuración (`config_loader.py`) leerá automáticamente estos cambios al iniciar la aplicación.

## Componentes Principales

- **`app.py`** — Entrypoint único; orquesta la UI y el flujo de entrenamiento
- **`config/processor.py`** — Limpieza, imputación y codificación de datos
- **`config/ann_service.py`** — Red neuronal: preproceso, construcción, entrenamiento y evaluación
- **`config/visualizer.py`** — Gráficos e iteractivos
- **`config/config_loader.py`** — Lee `config/appsettings.json`