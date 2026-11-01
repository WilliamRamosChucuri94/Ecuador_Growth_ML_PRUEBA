
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 1. Cargar modelo entrenado en Colab
modelo = joblib.load("modelo_pib.joblib")

# Compatibilidad con imputadores serializados
def reparar(objeto):
    if isinstance(objeto, SimpleImputer):
        if not hasattr(objeto, "_fill_dtype"):
            objeto._fill_dtype = np.asarray(
                objeto.statistics_
            ).dtype

    elif isinstance(objeto, Pipeline):
        for _, paso in objeto.steps:
            reparar(paso)

    elif isinstance(objeto, ColumnTransformer):
        for _, transformador, _ in objeto.transformers_:
            if isinstance(transformador, (Pipeline, SimpleImputer)):
                reparar(transformador)

reparar(modelo)

# 2. Nuevos escenarios hipotéticos
escenarios = pd.DataFrame({
    "petroleo": [2, -3, 5],
    "consumo": [3, 1, 4],
    "credito": [5, 2, 8],
    "exportaciones": [4, -2, 6]
})

# 3. Predecir crecimiento del PIB
variables = [
    "petroleo",
    "consumo",
    "credito",
    "exportaciones"
]

escenarios["prediccion_pib"] = modelo.predict(
    escenarios[variables]
)

# 4. Guardar resultados
escenarios.to_csv(
    "predicciones.csv",
    index=False
)

print("Predicciones generadas correctamente")
print(escenarios)
