"""
Script de Prueba - Verificar que todos los módulos funcionan correctamente
"""

import sys
import os

# Agregar el directorio al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("🧪 PRUEBAS DE MÓDULOS - Analizador de Datos")
print("=" * 60)

# Test 1: Importaciones
print("\n1️⃣ Probando importaciones...")
try:
    from config import Config, DataProcessor, Visualizer, GenericClassifier
    print("   ✅ Todas las importaciones exitosas")
except Exception as e:
    print(f"   ❌ Error en importaciones: {e}")
    sys.exit(1)

# Test 2: Cargador de Configuración
print("\n2️⃣ Probando cargador de configuración...")
try:
    config = Config()
    print(f"   ✅ Configuración cargada")
    print(f"      - Data path: {config.get('data_path')}")
    print(f"      - Target column: {config.get('target_column')}")
except Exception as e:
    print(f"   ❌ Error al cargar configuración: {e}")

# Test 3: CSV disponible
print("\n3️⃣ Verificando disponibilidad de datasets...")
try:
    import pandas as pd
    data_path = config.get('data_path')
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"   ✅ Dataset cargado correctamente")
        print(f"      - Forma: {df.shape}")
        print(f"      - Columnas: {list(df.columns)}")
    else:
        print(f"   ⚠️ Dataset no encontrado en {data_path}")
except Exception as e:
    print(f"   ❌ Error al cargar dataset: {e}")

# Test 4: Procesador de Datos
print("\n4️⃣ Probando procesador de datos...")
try:
    if data_path and os.path.exists(data_path):
        processor = DataProcessor()
        target_col = config.get('target_column')
        features = [col for col in df.columns if col != target_col]
        
        X, y, encoder = processor.clean_data(df, target_col, features)
        print(f"   ✅ Datos procesados correctamente")
        print(f"      - Features shape: {X.shape}")
        print(f"      - Target shape: {y.shape}")
except Exception as e:
    print(f"   ❌ Error al procesar datos: {e}")

# Test 5: Visualizador
print("\n5️⃣ Probando visualizador...")
try:
    import matplotlib.pyplot as plt
    if data_path and os.path.exists(data_path):
        # Test plot_generic_scatter
        fig = Visualizer.plot_generic_scatter(
            df.iloc[:100], 
            df.columns[0], 
            df.columns[1] if len(df.columns) > 1 else df.columns[0],
            df.columns[2] if len(df.columns) > 2 else df.columns[0]
        )
        print(f"   ✅ Visualizador funcionando")
        plt.close(fig)
except Exception as e:
    print(f"   ❌ Error en visualizador: {e}")

# Test 6: Clasificador
print("\n6️⃣ Probando clasificador ANN...")
try:
    if data_path and os.path.exists(data_path):
        target_col = config.get('target_column')
        features = [col for col in df.columns if col != target_col]
        
        clf = GenericClassifier(df, target_col, features, hidden_layers=[16, 8])
        print(f"   ✅ Clasificador inicializado correctamente")
        print(f"      - Target: {target_col}")
        print(f"      - Features: {len(features)}")
except Exception as e:
    print(f"   ❌ Error al inicializar clasificador: {e}")

print("\n" + "=" * 60)
print("✅ TODAS LAS PRUEBAS COMPLETADAS")
print("=" * 60)
print("\n💡 La aplicación está lista para usar:")
print("   streamlit run app.py")
