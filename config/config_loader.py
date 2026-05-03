import json
import os

class Config:
    def __init__(self, path='config/appsettings.json'):
        # Obtener la ruta absoluta respecto a la raíz del proyecto
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
        return self.data.get(key, default)

if __name__ == "__main__":
    # Bloque de prueba para verificar la carga de configuración de forma independiente
    config = Config()
    print("Configuración cargada correctamente:")
    print(config.data)