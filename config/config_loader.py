"""Carga centralizada de configuracion del proyecto desde appsettings.json."""

import json
import os

class Config:
    def __init__(self, path='config/appsettings.json'):
        """Lee el JSON de configuracion desde la raiz del proyecto.

        El parametro `path` se conserva por compatibilidad, pero la ruta se
        resuelve de forma relativa a este modulo para que funcione igual desde
        Streamlit, ejecucion directa o PyInstaller.
        """

        # Obtener la ruta absoluta respecto a la raiz del proyecto.
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base_path, 'config', 'appsettings.json')
        
        try:
            with open(full_path, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo en {full_path}")
            self.data = {}
        except json.JSONDecodeError:
            print(f"Error: El archivo {full_path} no tiene un formato JSON válido")
            self.data = {}

    def get(self, key, default=None):
        """Devuelve un valor de configuracion o un valor por defecto."""

        return self.data.get(key, default)

if __name__ == "__main__":
    # Bloque de prueba para verificar la carga de configuracion de forma independiente.
    config = Config()
    print("Configuración cargada correctamente:")
    print(config.data)