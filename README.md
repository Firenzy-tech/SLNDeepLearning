# SLNDeepLearning

Este proyecto proporciona una infraestructura base para experimentos de Deep Learning y Machine Learning, integrando un sistema de configuración centralizado y una interfaz gráfica interactiva.

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

El proyecto utiliza **Streamlit** para la interfaz gráfica. Para iniciar el panel de control, ejecuta el siguiente comando desde la raíz del proyecto:

```bash
streamlit run main.py
```

Una vez ejecutado, el sistema abrirá automáticamente una ventana en tu navegador predeterminado (normalmente en `http://localhost:8501`) donde podrás ajustar los hiperparámetros e iniciar el entrenamiento.

## Configuración de Hiperparámetros

Puedes modificar los valores por defecto editando el archivo `config/appsettings.json`. El cargador de configuración (`config_loader.py`) leerá automáticamente estos cambios al iniciar la aplicación.