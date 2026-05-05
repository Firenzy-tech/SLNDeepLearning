# Guía de Arquitectura

## Visión General

El proyecto está organizado como un panel de análisis de datos y entrenamiento de un clasificador binario con una red neuronal artificial. La entrada principal es [app.py](app.py), que arma la interfaz de Streamlit y conecta todas las demás piezas.

## Flujo de Ejecución

1. **Punto de entrada:** `streamlit run app.py`
2. `app.py` lee la configuración desde [config/appsettings.json](config/appsettings.json).
3. El usuario carga un CSV o usa el dataset de ejemplo definido en la configuración.
4. `DataProcessor` en [config/processor.py](config/processor.py) limpia los datos, imputa nulos y codifica variables categóricas.
5. `GenericClassifier` en [config/ann_service.py](config/ann_service.py) divide el dataset, escala las variables, construye la red y ejecuta el entrenamiento.
6. `Visualizer` en [config/visualizer.py](config/visualizer.py) genera gráficos de exploración, correlación, pérdida y matriz de confusión.
7. Cuando el entrenamiento termina, la app empaqueta el modelo con `save_assets()` para descargarlo como ZIP.

## Capas Del Proyecto

### Presentación

La capa de presentación está en `app.py`. Su trabajo es coordinar widgets, tabs, métricas, gráficos y acciones de usuario. También mantiene el estado de sesión de Streamlit para no repetir el entrenamiento ni el preprocesamiento en cada interacción.

### Configuración

`config/config_loader.py` carga `config/appsettings.json`. Ahí viven la ruta del dataset de ejemplo, la columna objetivo y los parámetros base del modelo clásico.

### Datos

`config/processor.py` se encarga del tratamiento básico de datos tabulares:

- elimina filas sin target
- rellena nulos numéricos con mediana
- rellena nulos categóricos con moda
- aplica one-hot encoding
- codifica el target si viene como texto

### Modelo ANN

`config/ann_service.py` implementa `GenericClassifier`. Esta clase encapsula el pipeline de machine learning usado por la app:

- `preprocess_data()` prepara train/test y escala
- `build_model()` define la arquitectura de la red
- `train()` ejecuta el entrenamiento
- `evaluate()` calcula métricas de clasificación
- `save_assets()` exporta el modelo y sus metadatos

### Visualización

`config/visualizer.py` centraliza las figuras para que la UI no mezcle lógica de negocio con lógica de gráficos.

## Librerías Principales

- `streamlit`: interfaz web interactiva.
- `pandas`: manipulación de datasets tabulares.
- `numpy`: soporte numérico general.
- `scikit-learn`: imputación, escalado, partición y métricas.
- `tensorflow`: red neuronal artificial.
- `matplotlib` y `seaborn`: gráficos estáticos.
- `plotly`: gráficos interactivos en la UI.

## Puntos A Tener En Cuenta

- El panel principal está pensado para clasificación binaria.
- El target debe tener exactamente dos clases para poder entrenar.
- El modelo exporta dos artefactos: el `.keras` y un `.pkl` con escalador y metadatos.
- El comando correcto para arrancar la app es `streamlit run app.py`.