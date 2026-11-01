from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Ecuador Growth ML · Simulado',page_icon='📈',layout='wide')
FEATURES = ['petroleo','consumo','credito','exportaciones']
@st.cache_resource
def load_model():
    return joblib.load(ROOT/'modelos'/'modelo_pib.joblib')

@st.cache_data
def load_data():
    return pd.read_csv(ROOT/'datos'/'crecimiento_simulado.csv'),pd.read_csv(ROOT/'datos'/'monitoreo_simulado.csv'),json.loads((ROOT/'modelos'/'metricas.json').read_text(encoding='utf-8'))

st.title('📈 Ecuador Growth ML')
st.warning('PROTOTIPO DIDÁCTICO: datos y resultados completamente SIMULADOS. No son cifras, estimaciones ni proyecciones oficiales del Ecuador.')
try:
    modelo=load_model(); datos,monitoreo,metricas=load_data()
except FileNotFoundError:
    st.error('Faltan archivos del modelo. Ejecuta primero: python generar_datos.py y python entrenar.py')
    st.stop()

p1,p2,p3=st.tabs(['🔮 Escenarios','📊 Datos y evaluación','🩺 Monitoreo'])
with p1:
    st.subheader('Simular escenario de crecimiento del PIB')
    st.caption('Tasas porcentuales hipotéticas. La predicción es asociativa, NO una estimación causal.')
    c1,c2=st.columns(2)
    with c1:
        petroleo=st.slider('Variación de producción petrolera (%)',-20.0,20.0,2.0,0.5)
        consumo=st.slider('Crecimiento del consumo (%)',-15.0,15.0,3.0,0.5)
    with c2:
        credito=st.slider('Crecimiento del crédito (%)',-20.0,25.0,5.0,0.5)
        exportaciones=st.slider('Crecimiento de exportaciones (%)',-20.0,25.0,4.0,0.5)
    escenario=pd.DataFrame([{'petroleo':petroleo,'consumo':consumo,'credito':credito,'exportaciones':exportaciones}])
    pronostico=float(modelo.predict(escenario[FEATURES])[0])
    st.metric('Crecimiento del PIB simulado',f'{pronostico:.2f}%')
    st.info(f"Modelo utilizado: {metricas['modelo_seleccionado']} · MAE fuera de muestra: {metricas['mae_prueba']:.2f} puntos porcentuales (sobre datos sintéticos).")
    st.caption('Modificar una entrada NO implica un efecto causal ni garantiza que la combinación sea económicamente plausible.')
with p2:
    st.subheader('Series trimestrales simuladas')
    st.line_chart(datos.set_index('trimestre')['crecimiento_pib'])
    st.subheader('Selección del modelo por validación cruzada temporal')
    st.dataframe(pd.DataFrame(metricas['comparacion_cv']),hide_index=True)
    col1,col2=st.columns(2)
    col1.metric('MAE prueba',f"{metricas['mae_prueba']:.2f} pp")
    col2.metric('RMSE prueba',f"{metricas['rmse_prueba']:.2f} pp")
    st.caption('Se selecciona por MAE medio de la validación cruzada sobre el 80% inicial; el 20% final se reserva para prueba.')
    st.download_button('Descargar datos simulados',datos.to_csv(index=False),'crecimiento_simulado.csv','text/csv')
with p3:
    st.subheader('Predicción versus dato observado (simulados)')
    st.line_chart(monitoreo.set_index('trimestre')[['crecimiento_pib','prediccion']])
    st.metric('MAE del periodo de prueba',f"{monitoreo['error_absoluto'].mean():.2f} pp")
    st.dataframe(monitoreo.tail(12),hide_index=True)
    st.caption('El monitoreo real exigiría incorporar observaciones posteriores y controlar cambios en las variables, la calidad de datos y el desempeño.')
