"""
Módulo para generar diagnósticos de modelos usando Groq API.
Genera explicaciones en lenguaje no técnico para personas sin conocimiento técnico.
"""

import requests
import os
import json
from typing import Dict, Any


class GroqDiagnostician:
    """Genera diagnósticos de modelos usando Groq, en lenguaje accesible."""
    
    def __init__(self, api_key: str = None):
        """
        Inicializa el cliente de Groq.
        
        Args:
            api_key: Token de API de Groq. Si no se proporciona, intenta leer de env var.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError(
                "❌ API Key de Groq no encontrada. "
                "Define la variable de entorno GROQ_API_KEY o pásala como parámetro."
            )
    
    def generate_diagnostic(self, metrics: Dict[str, Any], model_name: str = "Modelo de Clasificación") -> str:
        """
        Genera un diagnóstico no técnico del modelo basado en sus métricas.
        
        Args:
            metrics: Diccionario con las métricas del modelo:
                - accuracy: Precisión general (0-1)
                - precision: Precisión ponderada (0-1)
                - recall: Recall ponderado (0-1)
                - f1_score: F1-Score (0-1)
                - matrix_data: Datos de la matriz de confusión (optional)
            model_name: Nombre descriptivo del modelo
        
        Returns:
            Diagnóstico en texto explicado para personas no técnicas
        """
        
        # Formatea las métricas para legibilidad
        accuracy_pct = metrics.get('accuracy', 0) * 100
        precision_pct = metrics.get('precision', 0) * 100
        f1_pct = metrics.get('f1_score', 0) * 100
        
        # Prompt actualizado: solicita un análisis técnico (científico de datos)
        # seguido de un resumen claro para un administrativo.
        prompt = f"""Eres un científico de datos. Escribe en dos secciones claras:

    PARTE A — ANÁLISIS TÉCNICO (para otros científicos de datos):
    - Describe con vocabulario técnico y conciso el rendimiento del modelo usando las métricas proporcionadas.
    - Señala posibles causas (p. ej. overfitting, desequilibrio de clases), limitaciones y riesgos.

    PARTE B — RESUMEN PARA ADMINISTRATIVO (no técnico):
    - Explica en lenguaje no técnico el veredicto y la confianza (ej.: "Recomendado / Revisar / No recomendado").
    - Incluye una recomendación operativa priorizada y un breve comentario sobre riesgo.

    MÉTRICAS DEL MODELO "{model_name}":
    - Precisión General: {accuracy_pct:.1f}%
    - Precisión: {precision_pct:.1f}%
    - F1-Score: {f1_pct:.1f}%

    REQUISITOS:
    usa términos técnicos apropiados y evidencia numérica breve.
    lenguaje claro, una línea de veredicto y una recomendación práctica.
    Mantén cada parte concisa  Usa % cuando sea relevante.

    Finalmente evalua retorna cuales son las variables más importantes para el modelo y su impacto en el resultado, basándote en la matriz de confusión y las métricas proporcionadas.  Si no se proporcionan datos de la matriz de confusión, haz una inferencia basada en las métricas disponibles.

    
    """
        
        try:
            response = requests.post(
                self.api_url,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    "model": "openai/gpt-oss-20b",  # Modelo activo y rápido de Groq
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                timeout=60
            )
            
            if response.status_code == 200:
                diagnostic = response.json()['choices'][0]['message']['content']
                return diagnostic.strip()
            else:
                error_msg = response.json().get('error', {}).get('message', 'Error desconocido')
                raise Exception(f"Error de Groq: {error_msg}")
        
        except requests.exceptions.Timeout:
            raise Exception("⏱️ Timeout: La API de Groq tardó demasiado en responder.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"🌐 Error de conexión: {str(e)}")
    
    def validate_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Valida que las métricas contengan los campos necesarios."""
        required_fields = ['accuracy', 'precision', 'f1_score']
        return all(field in metrics for field in required_fields)
