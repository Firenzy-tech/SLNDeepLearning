#Manejo de logs de procesamiento para auditoría y depuración.
import logging
import os

def get_logger(name: str) -> logging.Logger:
    """Configura y retorna un logger estandarizado."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Consola
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger