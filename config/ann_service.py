import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.regularizers import l2

class GenericClassifier:
    def __init__(self, df, target_column, feature_columns, hidden_layers=[16, 8, 20], dropout_rate=0.2, l2_reg=0.001):
        """Clasificador dinámico basado en ANN."""
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

    def preprocess_data(self, test_size=0.2):
        """Implementa pasos automáticos de limpieza y escalado."""
        df_clean = self.df.dropna(subset=[self.target_column])
        X = df_clean[self.feature_columns].copy()
        y = df_clean[self.target_column].copy()

        # Codificación del Target
        if y.dtype == 'object' or y.dtype == 'category':
            y = self.label_encoder.fit_transform(y)

        # Imputación Automática
        num_cols = X.select_dtypes(include=['number']).columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(num_cols) > 0:
            X[num_cols] = self.imputer_num.fit_transform(X[num_cols])
        if len(cat_cols) > 0:
            X[cat_cols] = self.imputer_cat.fit_transform(X[cat_cols])
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

        # División y Escalado
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        self.input_dim = self.X_train.shape[1]
        return X.columns.tolist()

    def build_model(self, learning_rate=0.001):
        """Construye arquitectura flexible basada en parámetros."""
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

    def train(self, epochs=100, batch_size=10):
        if self.model is None: self.build_model()
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        return self.history

    def evaluate(self):
        """Genera métricas automáticas."""
        y_prob = self.model.predict(self.X_test, verbose=0)
        y_pred = (y_prob > 0.5).astype(int).flatten()
        cm = confusion_matrix(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        return cm, report, y_pred

    def save_assets(self, filename='ann_model'):
        """Exporta modelo y escalador."""
        self.model.save(f"{filename}.keras")
        assets = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'features': self.feature_columns
        }
        with open(f"{filename}_assets.pkl", 'wb') as f:
            pickle.dump(assets, f)