# 🧠 Analizador de Datos y Clasificador - Guía Completa

## 📋 Descripción General

Este proyecto proporciona una aplicación web interactiva construida con **Streamlit** para:
- ✅ Explorar y analizar datasets en formato CSV
- ✅ Visualizar correlaciones y distribuciones de datos
- ✅ Entrenar redes neuronales artificiales (ANN) personalizables
- ✅ Realizar predicciones en tiempo real

## 🚀 Instalación Rápida

### Requisitos
- Python 3.8 o superior
- pip (gestor de paquetes)

### Pasos

1. **Acceder al directorio del proyecto:**
```bash
cd c:\Users\Juan\source\repos\SLNDeepLearning
```

2. **Crear entorno virtual (recomendado):**
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **Ejecutar la aplicación:**
```bash
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`

## 📊 Características Principales

### 🔍 Pestaña Exploración
Análisis inicial de tus datos:
- Vista previa del dataset completo
- Conteo de valores nulos por columna
- Información técnica (tipos de datos, memoria)
- Vista de datos procesados y limpios

### 📈 Pestaña Visualización
Gráficas interactivas para análisis profundo:
- **Matriz de Correlaciones**: Identifica relaciones entre variables
- **Correlación con Objetivo**: Variables más influyentes
- **Distribución de Variables**: Histogramas con análisis de outliers
- **Gráficos de Dispersión**: Relaciones entre pares de variables
- **Análisis KDE**: Densidad de probabilidad

### 🧠 Pestaña Entrenamiento
Entrena tu modelo de red neuronal:
- Configuración flexible de arquitectura
- Monitoreo de progreso en tiempo real
- Gráficas de precisión y pérdida
- Matriz de confusión
- Predicciones sobre dataset de test

## ⚙️ Configuración

### Archivo: `config/appsettings.json`

```json
{
  "data_path": "dataSets/Data/weather_forecast_data.csv",
  "target_column": "Temperature",
  "model_params": {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
  }
}
```

**Opciones:**
- `data_path`: Ruta del CSV a cargar automáticamente
- `target_column`: Columna a predecir
- Deja vacío `data_path` para usar carga manual

## 🛠️ Parámetros del Modelo

### Parámetros Ajustables en la Interfaz

| Parámetro | Rango | Descripción |
|-----------|-------|------------|
| **Columna Objetivo** | - | Variable a predecir |
| **Variables de Entrada** | - | Features para entrenar |
| **Capas Ocultas** | - | Ej: "16, 8, 20" (3 capas) |
| **Dropout Rate** | 0.0 - 0.5 | Regularización por dropout |
| **Regularización L2** | 0.0001 - 0.1 | Control de overfitting |
| **Learning Rate** | 0.0001 - 1.0 | Velocidad de aprendizaje |
| **Épocas** | 10 - 500 | Iteraciones de entrenamiento |
| **Batch Size** | 1 - 128 | Muestras por iteración |

## 📂 Estructura del Proyecto

```
SLNDeepLearning/
├── app.py                    # Aplicación principal de Streamlit
├── requirements.txt          # Dependencias
├── config/
│   ├── __init__.py          # Importaciones del módulo
│   ├── appsettings.json     # Configuración global
│   ├── config_loader.py     # Cargador de configuración
│   ├── processor.py         # Procesamiento de datos
│   ├── visualizer.py        # Generador de gráficas
│   ├── ann_service.py       # Servicio de Red Neuronal
│   └── model_service.py     # Servicio de modelo
├── dataSets/
│   └── Data/
│       ├── weather_forecast_data.csv
│       └── mental_health.csv
└── build/                    # Compilados (PyInstaller)
```

## 🔄 Flujo de Trabajo

1. **Cargar Datos**
   - Automático desde `appsettings.json` o
   - Carga manual desde la interfaz

2. **Explorar Datos**
   - Revisar estadísticas en "Exploración"
   - Identificar problemas o anomalías

3. **Visualizar**
   - Analizar correlaciones
   - Identificar patrones
   - Seleccionar features relevantes

4. **Entrenar Modelo**
   - Ajustar parámetros en la barra lateral
   - Presionar "Iniciar Entrenamiento"
   - Monitorear métricas en tiempo real

5. **Evaluar & Predecir**
   - Ver matriz de confusión
   - Ejecutar predicciones sobre test set
   - Guardar modelo si es satisfactorio

## 💾 Guardar Modelo

Después del entrenamiento:
1. Presiona "💾 Guardar Activos del Modelo"
2. Los archivos se guardan en:
   - `ann_classifier_output.keras` (modelo)
   - `ann_classifier_output_assets.pkl` (escalador y encoder)

## 🐛 Solución de Problemas

### Las gráficas no se muestran
- ✅ Verificar backend de matplotlib: `matplotlib.use('Agg')`
- ✅ Asegurar que `plt.close()` se llama después de cada gráfico

### Error: "cannot do a non-empty take from an empty axes"
- ✅ Seleccionar variables con suficiente varianza
- ✅ Evitar seleccionar la misma variable en X e Y

### Entrenamiento muy lento
- ✅ Reducir número de épocas
- ✅ Aumentar batch size
- ✅ Reducir tamaño del dataset

## 📦 Dependencias Principales

```
streamlit          - Framework web interactivo
pandas            - Manipulación de datos
numpy             - Computación numérica
scikit-learn      - Machine Learning
tensorflow        - Deep Learning
matplotlib        - Visualización
seaborn           - Gráficas estadísticas
```

## ✨ Mejoras Recientes

- ✅ Backend de matplotlib configurado correctamente
- ✅ Cierre adecuado de figuras matplotlib
- ✅ Validación robusta de datos
- ✅ Manejo de excepciones en visualización
- ✅ Datos precargados para demostración
- ✅ Todas las gráficas funcionando correctamente

## 📞 Soporte

Para reportar problemas o sugerencias, revisa:
- Archivo `MEJORAS.md` - Historial de correcciones
- Archivo `diagnose.py` - Script de diagnóstico

## 📝 Licencia

[Especificar licencia del proyecto]

---

**Versión**: 2.0
**Última actualización**: Mayo 2026
