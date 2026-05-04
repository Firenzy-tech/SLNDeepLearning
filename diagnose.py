import pandas as pd
import os

# Verificar que el CSV existe
csv_path = "dataSets/Data/weather_forecast_data.csv"
if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ CSV cargado correctamente")
        print(f"Dimensiones: {df.shape}")
        print(f"Columnas: {list(df.columns)}")
        print(f"\nPrimeras 3 filas:")
        print(df.head(3))
        print(f"\nTipos de datos:")
        print(df.dtypes)
    except Exception as e:
        print(f"❌ Error al leer CSV: {e}")
else:
    print(f"❌ Archivo no encontrado: {csv_path}")

# Verificar configuración
try:
    from config import Config
    config = Config()
    print(f"\n✅ Configuración cargada")
    print(f"Data path: {config.get('data_path')}")
    print(f"Target column: {config.get('target_column')}")
except Exception as e:
    print(f"❌ Error al cargar configuración: {e}")
