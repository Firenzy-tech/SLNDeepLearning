#Abstracción de Scikit-Learn.

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import pandas as pd
from typing import Dict, Any, Tuple

def run_ml_pipeline(df: pd.DataFrame, target_col: str, task: str) -> Tuple[Any, Dict[str, Any]]:
    """Ejecuta un pipeline básico de Machine Learning."""
    # Separación de características y objetivo
    X = df.drop(columns=[target_col])
    # Asegurar que X sea numérico para este ejemplo básico
    X = X.select_dtypes(include=['number']) 
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task == "Clasificación":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "Accuracy": accuracy_score(y_test, preds),
            "Report": classification_report(y_test, preds, output_dict=True)
        }
    else: # Regresión
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "MSE": mean_squared_error(y_test, preds)
        }
        
    return model, metrics