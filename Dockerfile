# 1. Imagen base oficial de Python
FROM python:3.9-slim-bookworm

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Instalar dependencias del sistema necesarias para algunas librerías de ML
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copiar e instalar las dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el resto del código del proyecto
COPY . .

# 6. Exponer el puerto predeterminado de Streamlit
EXPOSE 8501

# 7. Configuración de salud (Healthcheck) para verificar que la app responda
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 8. Comando para ejecutar la aplicación
# Usamos --server.address=0.0.0.0 para que sea accesible fuera del contenedor
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]