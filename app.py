
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

# ==========================================================
# 1. CONFIGURACIÓN
# ==========================================================

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Ecuador Growth ML · Simulado",
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
# 2. CARGAR MODELO Y DATOS
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "modelo_pib.joblib")


@st.cache_data
def load_data():
    datos = pd.read_csv(ROOT / "crecimiento_simulado.csv")

    monitoreo = pd.read_csv(ROOT / "monitoreo_simulado.csv")

    with open(ROOT / "metricas.json", encoding="utf-8") as f:
        metricas = json.load(f)

    return datos, monitoreo, metricas


# ==========================================================
# 3. PRESENTACIÓN
# ==========================================================

st.title("📈 Ecuador Growth ML")

st.warning(
    "PROTOTIPO DIDÁCTICO: datos y resultados completamente "
    "SIMULADOS. No son cifras ni proyecciones oficiales del Ecuador."
)

try:
    modelo = load_model()
    datos, monitoreo, metricas = load_data()

except FileNotFoundError as e:
    st.error(f"No se encontró el archivo: {e.filename}")
    st.stop()


# ==========================================================
# 4. PESTAÑAS
# ==========================================================

p1, p2, p3 = st.tabs([
    "🔮 Escenarios",
    "📊 Datos y evaluación",
    "🩺 Monitoreo"
])


# ==========================================================
# 5. SIMULACIÓN DE ESCENARIOS
# ==========================================================

with p1:

    st.subheader("Simular el crecimiento económico")

    st.caption(
        "Modificar las variables permite explorar escenarios "
        "hipotéticos, no estimar efectos causales."
    )

    c1, c2 = st.columns(2)

    with c1:
        petroleo = st.slider(
            "Producción petrolera (%)",
            -20.0, 20.0, 2.0, 0.5
        )

        consumo = st.slider(
            "Crecimiento del consumo (%)",
            -15.0, 15.0, 3.0, 0.5
        )

    with c2:
        credito = st.slider(
            "Crecimiento del crédito (%)",
            -20.0, 25.0, 5.0, 0.5
        )

        exportaciones = st.slider(
            "Crecimiento de exportaciones (%)",
            -20.0, 25.0, 4.0, 0.5
        )

    escenario = pd.DataFrame([{
        "petroleo": petroleo,
        "consumo": consumo,
        "credito": credito,
        "exportaciones": exportaciones
    }])

    prediccion = modelo.predict(escenario[FEATURES])[0]

    st.metric(
        "Crecimiento del PIB simulado",
        f"{prediccion:.2f}%"
    )

    st.info(
        f"Modelo: {metricas['modelo_seleccionado']} | "
        f"MAE de prueba: {metricas['mae_prueba']:.2f} pp"
    )


# ==========================================================
# 6. DATOS Y EVALUACIÓN
# ==========================================================

with p2:

    st.subheader("Crecimiento económico trimestral simulado")

    st.line_chart(
        datos.set_index("trimestre")["crecimiento_pib"]
    )

    st.subheader("Comparación de modelos")

    st.dataframe(
        pd.DataFrame(metricas["comparacion_cv"]),
        hide_index=True
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "MAE",
        f"{metricas['mae_prueba']:.2f} pp"
    )

    c2.metric(
        "RMSE",
        f"{metricas['rmse_prueba']:.2f} pp"
    )

    st.download_button(
        "Descargar datos simulados",
        datos.to_csv(index=False),
        "crecimiento_simulado.csv",
        "text/csv"
    )


# ==========================================================
# 7. MONITOREO DEL MODELO
# ==========================================================

with p3:

    st.subheader("Predicción vs crecimiento observado")

    st.line_chart(
        monitoreo.set_index("trimestre")[
            ["crecimiento_pib", "prediccion"]
        ]
    )

    mae = monitoreo["error_absoluto"].mean()

    st.metric(
        "Error absoluto medio",
        f"{mae:.2f} pp"
    )

    st.dataframe(
        monitoreo.tail(12),
        hide_index=True
    )

    st.caption(
        "Monitoreo ilustrativo calculado con datos simulados. "
        "En producción se incorporarían nuevas observaciones."
    )
