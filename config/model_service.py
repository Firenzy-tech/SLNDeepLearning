from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ModelService:
    def __init__(self, model_params):
        # Inicializamos un clasificador genérico (ej. RandomForest) con los parámetros del JSON
        self.model = RandomForestClassifier(**model_params)
        self.is_trained = False

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        self.is_trained = True
        return accuracy

    def predict(self, X):
        if not self.is_trained:
            raise Exception("El modelo debe ser entrenado antes de realizar inferencias.")
        return self.model.predict(X)