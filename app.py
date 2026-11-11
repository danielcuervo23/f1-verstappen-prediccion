import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import load_data, preprocessing, train, evaluate, predict

st.title('Predicción de la posición final de Max Verstappen')
df = load_data()
st.subheader('Dataset Preview')
st.dataframe(df.head())

X, y = preprocessing(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
name = st.selectbox('Modelo', ['Regresión Lineal', 'Árbol de Decisión', 'Random Forest'])
model = train(X_train, y_train, name)

st.subheader('Datos de la carrera')
year = st.selectbox('Año', sorted(df['Year'].unique()))
location = st.selectbox('Circuito', sorted(df['Location'].unique()))
weather = st.selectbox('Clima', sorted(df['Weather'].unique()))
quali = st.slider('Posición de clasificación', 1, 20, 3)
start = st.slider('Posición de salida', 1, 20, 3)

pred = predict(model, X.columns, year, location, weather, quali, start)
st.metric('Posición final predicha', pred)

fig, ax = plt.subplots()
ax.bar(['Salida', 'Predicción'], [start, pred])
ax.set_ylabel('Posición')
st.pyplot(fig)

mae, mse, rmse, r2 = evaluate(y_test, model.predict(X_test))
st.subheader('Métricas en prueba')
st.write('MAE', mae)
st.write('MSE', mse)
st.write('RMSE', rmse)
st.write('R2', r2)
mae_simple, _, _, _ = evaluate(y_test, X_test['StartPosition'])
st.write('Regla simple "termina donde salió" (MAE en prueba)', mae_simple)
