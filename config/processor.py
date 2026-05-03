import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    @staticmethod
    def clean_data(df, target_column, selected_features=None):
        """
        Realiza una limpieza básica: maneja valores nulos y codifica variables categóricas.
        """
        # Trabajamos con una copia para evitar SettingWithCopyWarning
        df = df.copy()

        # Eliminar filas donde el objetivo es nulo
        df = df.dropna(subset=[target_column])
        
        # Verificar si el dataset quedó vacío después de la limpieza inicial
        if df.empty:
            raise ValueError(f"El dataset está vacío tras eliminar los valores nulos de la columna objetivo '{target_column}'. Por favor, revisa tus datos.")

        # Filtrar solo las columnas seleccionadas si se proporcionan
        if selected_features:
            df = df[list(selected_features) + [target_column]]

        # Separar target de features para el procesamiento
        target = df[target_column]
        features_df = df.drop(columns=[target_column])

        # Identificar columnas numéricas y categóricas en las features
        numeric_cols = features_df.select_dtypes(include=['number']).columns
        categorical_feature_cols = features_df.select_dtypes(include=['object', 'category']).columns
        
        # Imputar valores nulos
        for col in numeric_cols:
            features_df[col] = features_df[col].fillna(features_df[col].median())
        
        for col in categorical_feature_cols:
            mode_val = features_df[col].mode()
            features_df[col] = features_df[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")
            
        # Codificación One-Hot solo para las features categóricas
        features = pd.get_dummies(features_df, columns=categorical_feature_cols, drop_first=True)

        # Aplicar Label Encoding al target si es categórico
        label_encoder = None
        if target.dtype == 'object' or target.dtype == 'category':
            label_encoder = LabelEncoder()
            target = label_encoder.fit_transform(target)
            # st.session_state['target_label_encoder'] = label_encoder # Store for potential inverse transform or display

        return features, target, label_encoder