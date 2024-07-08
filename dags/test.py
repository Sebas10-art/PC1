import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from automl import MLSystem  # Ajusta según el nombre de tu archivo y clase

class TestMLSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Cargar el modelo preentrenado desde .sav para todas las pruebas
        cls.model_path = 'final_dt_.sav'
        cls.model = joblib.load(cls.model_path)

    def setUp(self):
        # Inicializar una nueva instancia de MLSystem para cada prueba
        self.ml_system = MLSystem()

        # Cargar datos de entrenamiento para cada prueba
        self.ml_system.load_data()

    def test_train_model(self):
        # Asignar el modelo cargado para entrenar
        self.ml_system.model = self.model
        self.ml_system.train_model()

        # Verificar que el modelo esté entrenado correctamente
        self.assertTrue(hasattr(self.ml_system, 'model'))
        self.assertIsNotNone(self.ml_system.model)

    def test_evaluate_model(self):
        # Asignar el modelo cargado para evaluar
        self.ml_system.model = self.model
        self.ml_system.train_model()
        accuracy = self.ml_system.evaluate_model()

        # Verificar que la precisión sea mayor que 0.5 (ejemplo)
        self.assertGreater(accuracy, 0.5)

    def test_predict(self):
        # Preparar datos de ejemplo para hacer predicciones
        X_sample = self.ml_system.X_test[:10]  # Tomar las primeras 10 filas como ejemplo

        # Asignar el modelo cargado para hacer predicciones
        self.ml_system.model = self.model
        self.ml_system.train_model()
        y_pred = self.ml_system.predict(X_sample)

        # Verificar que las predicciones tengan la forma esperada
        self.assertEqual(len(y_pred), len(X_sample))

if __name__ == '__main__':
    unittest.main()