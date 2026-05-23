"""Servicio de clasificacion neuronal para el flujo principal de la app.

La clase `GenericClassifier` concentra todo el pipeline de ML:
- limpia y prepara los datos
- separa train/test y escala features
- construye la ANN configurable
- entrena y evalua el modelo
- exporta modelo + artefactos auxiliares
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.regularizers import l2

class GenericClassifier:
    def __init__(self, df, target_column, feature_columns, hidden_layers=[16, 8, 20], dropout_rate=0.2, l2_reg=0.001):
        """Inicializa el clasificador y los transformadores auxiliares."""

        self.df = df.copy()
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        self.history = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.processed_features = None

    def preprocess_data(self, test_size=0.2):
        """Aplica limpieza, codificacion, particion y escalado.

        Devuelve el listado final de columnas resultantes tras el one-hot encoding.
        """

        df_clean = self.df.dropna(subset=[self.target_column])
        X = df_clean[self.feature_columns].copy()
        y = df_clean[self.target_column].copy()

        # Si el target viene como texto, lo pasamos a etiquetas numericas.
        if y.dtype == 'object' or y.dtype == 'category':
            y = self.label_encoder.fit_transform(y)

        # Imputacion automatica para no depender de limpieza previa perfecta.
        num_cols = X.select_dtypes(include=['number']).columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(num_cols) > 0:
            X[num_cols] = self.imputer_num.fit_transform(X[num_cols])
        if len(cat_cols) > 0:
            X[cat_cols] = self.imputer_cat.fit_transform(X[cat_cols])
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

        # Division train/test y escalado para estabilizar el entrenamiento de la red.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        self.input_dim = self.X_train.shape[1]
        self.processed_features = X.columns.tolist()
        return self.processed_features

    def build_model(self, learning_rate=0.001):
        """Construye la ANN con las capas ocultas definidas por el usuario."""
        # Asegurarse de que los datos han sido preprocesados antes de construir
        if not hasattr(self, 'input_dim') or self.input_dim is None:
            raise ValueError("Ejecuta `preprocess_data()` antes de construir el modelo; no hay `input_dim` definido.")
        model = Sequential()
        model.add(Input(shape=(self.input_dim,)))
        for units in self.hidden_layers:
            model.add(Dense(units=units, activation='relu', kernel_regularizer=l2(self.l2_reg)))
            if self.dropout_rate > 0:
                model.add(Dropout(self.dropout_rate))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
        return self.model

    def train(self, epochs=100, batch_size=10, callbacks=None):
        """Entrena el modelo usando una validacion interna sobre el train set."""
        # Requerir preprocesado previo para evitar errores silenciosos
        if self.X_train is None:
            raise ValueError("Los datos no han sido preprocesados. Ejecuta `preprocess_data()` antes de entrenar.")

        if self.model is None:
            self.build_model()
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks
        )
        return self.history

    def evaluate(self):
        """Calcula predicciones, confusion matrix y classification report."""

        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado aún.")

        y_prob = self.model.predict(self.X_test, verbose=0)
        y_pred = (y_prob > 0.5).astype(int).flatten()
        cm = confusion_matrix(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        return cm, report, y_pred

    def explain_model(self, sample_size=50):
        """Calcula los valores SHAP para interpretar el modelo usando KernelExplainer."""
        if not SHAP_AVAILABLE:
            raise ImportError("La librería 'shap' no está instalada. Ejecuta 'pip install shap' para usar esta función.")

        if self.model is None:
            raise ValueError("El modelo debe estar entrenado antes de generar la explicación SHAP.")

        if self.X_test is None or len(self.X_test) == 0:
            raise ValueError("No hay datos de prueba disponibles para generar la explicación SHAP.")

        # Usar kmeans para resumir el background y que KernelExplainer corra rápido
        n_clusters = min(10, len(self.X_train))
        background = shap.kmeans(self.X_train, n_clusters)
        explainer = shap.KernelExplainer(self.model.predict, background)
        
        # Explicar un subset del test set para optimizar el rendimiento
        # Aseguramos no superar el límite del test set
        actual_size = min(sample_size, len(self.X_test))
        X_sample = self.X_test[:actual_size]
        
        shap_values = explainer.shap_values(X_sample)
        
        # En clasificación binaria Keras, shap_values puede venir como una lista de 1 elemento
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        return shap_values, X_sample, explainer.expected_value

    def save_assets(self, filename='ann_model'):
        """Exporta modelo, escalador y metadatos de features a disco."""

        if self.model is None:
            raise ValueError("El modelo no ha sido construido ni entrenado aún.")

        if self.processed_features is None:
            raise ValueError("Los datos no han sido preprocesados aún.")

        self.model.save(f"{filename}.keras")
        assets = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'features': self.processed_features # Guardar las features procesadas (incluyendo one-hot encoding)
        }
        with open(f"{filename}_assets.pkl", 'wb') as f:
            pickle.dump(assets, f)