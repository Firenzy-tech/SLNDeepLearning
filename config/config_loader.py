import json
import os

class Config:
    def __init__(self, path='config/appsettings.json'):
        with open(path, 'r') as f:
            self.data = json.load(f)
            
    def get(self, key, default=None):
        return self.data.get(key, default)

if __name__ == "__main__":
    # Bloque de prueba para verificar la carga de configuración de forma independiente
    config = Config()
    print("Configuración cargada correctamente:")
    print(config.data)