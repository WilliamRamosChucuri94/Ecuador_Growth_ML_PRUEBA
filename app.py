
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ==========================================================
# 1. CONFIGURACIÓN
# ==========================================================

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Ecuador Growth ML | Simulado",
    page_icon="📈",
    layout="wide"
)

FEATURES = [
    "petroleo",
    "consumo",
    "credito",
    "exportaciones"
]

# ==========================================================
# 2. CORRECCIÓN DE COMPATIBILIDAD DEL MODELO
# ==========================================================

def reparar_imputadores(objeto):
    """
    Corrección provisional para modelos guardados con una
    versión anterior de scikit-learn.

    El atributo interno _fill_dtype puede faltar al cargar
    ciertos SimpleImputer serializados.
    """

    if isinstance(objeto, SimpleImputer):

        if not hasattr(objeto, "_fill_dtype"):

            if hasattr(objeto, "statistics_"):
                objeto._fill_dtype = np.asarray(
                    objeto.statistics_
                ).dtype

    elif isinstance(objeto, Pipeline):

        for _, paso in objeto.steps:
            reparar_imputadores(paso)

    elif isinstance(objeto, ColumnTransformer):

        for _, transformador, _ in objeto.transformers_:

            if transformador not in ("drop", "passthrough"):
                reparar_imputadores(transformador)

    return objeto


# ==========================================================
# 3. CARGAR MODELO Y DATOS
# ==========================================================

@st.cache_resource
def load_model():

    ruta = ROOT / "modelo_pib.joblib"

    modelo = joblib.load(ruta)

    modelo = reparar_imputadores(modelo)

    return modelo


@st.cache_data
def load_data():

    datos = pd.read_csv(
        ROOT / "crecimiento_simulado.csv"
    )

    monitoreo = pd.read_csv(
        ROOT / "monitoreo_simulado.csv"
    )

    with open(
        ROOT / "metricas.json",
        encoding="utf-8"
    ) as archivo:

        metricas = json.load(archivo)

    return datos, monitoreo, metricas


# ==========================================================
# 4. PRESENTACIÓN
# ==========================================================

st.title("📈 Ecuador Growth ML")

st.warning(
    "PROTOTIPO DIDÁCTICO: datos y resultados completamente "
    "SIMULADOS. No son cifras ni proyecciones oficiales "
    "del Ecuador."
)

try:

    modelo = load_model()

    datos, monitoreo, metricas = load_data()

except FileNotFoundError as error:

    st.error(
        f"No se encontró el archivo: {error.filename}"
    )

    st.stop()

except Exception as error:

    st.error(
        "No fue posible cargar el modelo o los datos."
    )

    st.exception(error)

    st.stop()


# ==========================================================
# 5. PESTAÑAS DE LA APLICACIÓN
# ==========================================================

p1, p2, p3 = st.tabs([
    "🔮 Escenarios",
    "📊 Datos y evaluación",
    "🩺 Monitoreo"
])


# ==========================================================
# 6. SIMULACIÓN DE ESCENARIOS
# ==========================================================

with p1:

    st.subheader(
        "Simulador del crecimiento económico del Ecuador"
    )

    st.caption(
        "Modifica los indicadores económicos para explorar "
        "escenarios hipotéticos. Los resultados no representan "
        "relaciones causales."
    )

    c1, c2 = st.columns(2)

    with c1:

        petroleo = st.slider(
            "Producción petrolera (%)",
            min_value=-20.0,
            max_value=20.0,
            value=2.0,
            step=0.5
        )

        consumo = st.slider(
            "Crecimiento del consumo (%)",
            min_value=-15.0,
            max_value=15.0,
            value=3.0,
            step=0.5
        )

    with c2:

        credito = st.slider(
            "Crecimiento del crédito (%)",
            min_value=-20.0,
            max_value=25.0,
            value=5.0,
            step=0.5
        )

        exportaciones = st.slider(
            "Crecimiento de exportaciones (%)",
            min_value=-20.0,
            max_value=25.0,
            value=4.0,
            step=0.5
        )

    # Variables explicativas
    escenario = pd.DataFrame([{
        "petroleo": petroleo,
        "consumo": consumo,
        "credito": credito,
        "exportaciones": exportaciones
    }])

    st.divider()

    # Predicción mediante el modelo de Machine Learning
    try:

        prediccion = float(
            modelo.predict(escenario[FEATURES])[0]
        )

        st.metric(
            label="Crecimiento del PIB simulado",
            value=f"{prediccion:.2f}%"
        )

        st.info(
            f"Modelo seleccionado: "
            f"{metricas['modelo_seleccionado']} | "
            f"MAE de prueba: "
            f"{metricas['mae_prueba']:.2f} pp"
        )

    except Exception as error:

        st.error(
            "No se pudo generar la predicción. "
            "Comprueba la compatibilidad del modelo."
        )

        st.exception(error)

    st.caption(
        "La simulación modifica entradas del modelo "
        "predictivo. No permite inferir efectos causales "
        "de las variables sobre el PIB."
    )


# ==========================================================
# 7. DATOS Y EVALUACIÓN DEL MODELO
# ==========================================================

with p2:

    st.subheader(
        "Evolución del crecimiento económico simulado"
    )

    st.line_chart(
        datos.set_index("trimestre")[
            "crecimiento_pib"
        ]
    )

    st.divider()

    st.subheader(
        "Comparación de modelos de Machine Learning"
    )

    comparacion = pd.DataFrame(
        metricas["comparacion_cv"]
    )

    st.dataframe(
        comparacion,
        hide_index=True,
        use_container_width=True
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "MAE de prueba",
            f"{metricas['mae_prueba']:.2f} pp"
        )

    with c2:

        st.metric(
            "RMSE de prueba",
            f"{metricas['rmse_prueba']:.2f} pp"
        )

    st.info(
        "El MAE mide el error absoluto promedio. "
        "El RMSE penaliza más los errores grandes. "
        "Ambos se expresan en puntos porcentuales."
    )

    st.download_button(
        label="Descargar base simulada",
        data=datos.to_csv(index=False),
        file_name="crecimiento_simulado.csv",
        mime="text/csv"
    )


# ==========================================================
# 8. MONITOREO DEL MODELO
# ==========================================================

with p3:

    st.subheader(
        "Monitoreo del crecimiento económico simulado"
    )

    st.caption(
        "Comparación entre los valores observados "
        "sintéticos y las predicciones del modelo."
    )

    st.line_chart(
        monitoreo.set_index("trimestre")[
            ["crecimiento_pib", "prediccion"]
        ]
    )

    st.divider()

    mae = monitoreo["error_absoluto"].mean()

    st.metric(
        "Error absoluto medio",
        f"{mae:.2f} pp"
    )

    st.subheader(
        "Últimas observaciones"
    )

    st.dataframe(
        monitoreo.tail(12),
        hide_index=True,
        use_container_width=True
    )

    st.download_button(
        label="Descargar resultados de monitoreo",
        data=monitoreo.to_csv(index=False),
        file_name="monitoreo_resultados.csv",
        mime="text/csv"
    )

    st.caption(
        "Este monitoreo utiliza datos simulados. "
        "En una aplicación real sería necesario "
        "incorporar información nueva para evaluar "
        "la estabilidad y precisión predictiva."
    )
