# Motor de limpieza. Implementa el patrón Strategy para aplicar diferentes técnicas.
import pandas as pd
from typing import List


class DataCleaner:
    """Servicio centralizado para la transformación y limpieza de datos."""

    @staticmethod
    def handle_missing(df: pd.DataFrame, strategy: str, columns: List[str]) -> pd.DataFrame:
        """Maneja valores nulos según la estrategia seleccionada."""
        df_clean = df.copy()
        if strategy == "Eliminar filas":
            df_clean = df_clean.dropna(subset=columns)
        elif strategy == "Imputar media":
            for col in columns:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif strategy == "Imputar mediana":
            for col in columns:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif strategy == "Forward fill":
            df_clean[columns] = df_clean[columns].ffill()
            
        return df_clean

    @staticmethod
    def encode_categorical(df: pd.DataFrame, columns: List[str], method: str) -> pd.DataFrame:
        """Codifica variables categóricas."""
        df_clean = df.copy()
        if method == "Label Encoding":
            try:
                from sklearn.preprocessing import LabelEncoder
            except Exception as e:
                raise RuntimeError("scikit-learn is required for Label Encoding. Install it with: pip install scikit-learn") from e
            le = LabelEncoder()
            for col in columns:
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        elif method == "One Hot Encoding":
            df_clean = pd.get_dummies(df_clean, columns=columns)
        return df_clean

    @staticmethod
    def scale_numerical(df: pd.DataFrame, columns: List[str], method: str) -> pd.DataFrame:
        """Escala variables numéricas."""
        df_clean = df.copy()
        try:
            from sklearn.preprocessing import MinMaxScaler, StandardScaler
        except Exception as e:
            raise RuntimeError("scikit-learn is required for scaling. Install it with: pip install scikit-learn") from e
        scaler = StandardScaler() if method == "Estandarización" else MinMaxScaler()
        df_clean[columns] = scaler.fit_transform(df_clean[columns])
        return df_clean