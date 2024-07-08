import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import unittest

import joblib

# Cargar el modelo desde el archivo .pkl
modelo = joblib.load('final_dt_.pkl')

# Guardar el modelo como archivo .sav
joblib.dump(modelo, 'final_dt_.sav')

modelo_c = joblib.load('final_dt_.sav')

class MLSystem:
    def __init__(self):
        self.model = modelo_c
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        df = pd.read_csv('train.csv')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df.drop(columns=["Target"]), df["Target"], test_size=0.3, random_state=123)

    def train_model(self):
        self.y_train_df = pd.DataFrame(self.y_train)
        self.model.fit(self.X_train, self.y_train_df)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)

    def predict(self, X):
        return self.model.predict(X)

# Ejemplo de uso si se ejecuta como script principal
if __name__ == '__main__':
    # Ejemplo básico de carga de datos y entrenamiento
    ml_system = MLSystem()
    ml_system.load_data()
    ml_system.model = modelo_c# Instanciar el modelo aquí
    ml_system.train_model()
    accuracy = ml_system.evaluate_model()
    print(f'Accuracy: {accuracy}')
