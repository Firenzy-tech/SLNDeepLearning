"""Preprocesamiento de datos para el clasificador binario.

La clase `DataProcessor` toma un DataFrame crudo y devuelve:
- features ya limpias y codificadas
- target convertido a valores numericos si hace falta
- un `LabelEncoder` opcional para revertir el target despues
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    @staticmethod
    def clean_data(df, target_column, selected_features=None):
        """
        Limpia el dataset y lo deja listo para entrenamiento.

        Pasos:
        1. Elimina filas sin target.
        2. Filtra solo las columnas elegidas por el usuario.
        3. Imputa nulos numericos con mediana.
        4. Imputa nulos categoricos con moda.
        5. Aplica one-hot encoding a features categoricas.
        6. Codifica el target si es texto/categoria.
        """
        # Trabajamos con una copia para evitar SettingWithCopyWarning
        df = df.copy()

        # Eliminar filas donde el objetivo es nulo
        df = df.dropna(subset=[target_column])
        
        # Verificar si el dataset quedó vacío después de la limpieza inicial
        if df.empty:
            raise ValueError(f"El dataset está vacío tras eliminar los valores nulos de la columna objetivo '{target_column}'. Por favor, revisa tus datos.")

        # Filtrar solo las columnas seleccionadas si se proporcionan.
        if selected_features:
            df = df[list(selected_features) + [target_column]]

        # Separar target de features para el procesamiento.
        target = df[target_column]
        features_df = df.drop(columns=[target_column])

        # Identificar columnas numericas y categoricas en las features.
        numeric_cols = features_df.select_dtypes(include=['number']).columns
        categorical_feature_cols = features_df.select_dtypes(include=['object', 'category']).columns
        
        # Imputacion simple para no romper el entrenamiento por valores faltantes.
        for col in numeric_cols:
            features_df[col] = features_df[col].fillna(features_df[col].median())
        
        for col in categorical_feature_cols:
            mode_val = features_df[col].mode()
            features_df[col] = features_df[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")
            
        # Codificacion one-hot solo para las features categoricas.
        features = pd.get_dummies(features_df, columns=categorical_feature_cols, drop_first=True)

        # Aplicar label encoding al target si es categorico.
        label_encoder = None
        if target.dtype == 'object' or target.dtype == 'category':
            label_encoder = LabelEncoder()
            target = label_encoder.fit_transform(target)

        return features, target, label_encoder