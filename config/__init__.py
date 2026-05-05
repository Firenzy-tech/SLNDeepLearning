"""Reexporta las piezas principales de la capa de configuracion y servicios.

Permite importar desde config como fachada simple:
from config import Config, DataProcessor, Visualizer
"""

from .config_loader import Config
from .processor import DataProcessor
from .model_service import ModelService
from .visualizer import Visualizer
from .ann_service import GenericClassifier

# Esto permite importar todo desde 'config' directamente
__all__ = ['Config', 'DataProcessor', 'ModelService', 'Visualizer', 'GenericClassifier']