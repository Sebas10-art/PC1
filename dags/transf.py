import joblib

# Cargar el modelo desde el archivo .sav
modelo_sav = joblib.load('final_dt_.sav')

# Guardar el modelo como archivo .joblib
joblib.dump(modelo_sav, 'final_dt_.joblib')

print("Modelo convertido y guardado exitosamente como final_dt_.joblib")