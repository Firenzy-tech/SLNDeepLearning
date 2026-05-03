from config.config_loader import Config

def main():
    # Inicializar la configuración global
    config = Config()
    
    # Ejemplo de acceso a los hiperparámetros definidos en appsettings.json
    model_config = config.get("model", {})
    hyperparameters = model_config.get("hyperparameters", {})
    learning_rate = hyperparameters.get("learning_rate", 0.001)
    
    print(f"--- Sistema de Deep Learning Iniciado ---")
    print(f"Configuración detectada: Learning Rate = {learning_rate}")

if __name__ == "__main__":
    main()