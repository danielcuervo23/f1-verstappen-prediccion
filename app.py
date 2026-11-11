import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import load_data, preprocessing, train, evaluate, predict

st.title('Predicción de la posición final de Max Verstappen')

df = load_data()

st.subheader('Vista previa del dataset')
st.dataframe(df.head())

X, y = preprocessing(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

featureMode = st.selectbox('Modo de variables', ['Selección manual', 'Reducción con PCA'])

if featureMode == 'Reducción con PCA':
    n_columns = X_train['Location'].nunique() + X_train['Weather'].nunique() + 3
    n_components = st.slider('Número de componentes PCA', 2, n_columns, 30)
else:
    n_components = None

modelSelection = st.selectbox('Seleccione el modelo', ['Regresión Lineal', 'Árbol de Decisión', 'Random Forest'], index=2)

if modelSelection == 'Regresión Lineal':
    model = train(X_train, y_train, modelSelection, n_components=n_components)

elif modelSelection == 'Árbol de Decisión':
    max_depth = st.slider('Profundidad máxima (max_depth)', 1, 20, 3)
    min_samples_leaf = st.slider('Mínimo de muestras por hoja (min_samples_leaf)', 1, 30, 20)

    model = train(X_train, y_train, modelSelection, max_depth=max_depth,
                  min_samples_leaf=min_samples_leaf, n_components=n_components)

elif modelSelection == 'Random Forest':
    n_estimators = st.slider('Número de árboles (n_estimators)', 10, 300, 100, 10)
    max_depth = st.slider('Profundidad máxima (max_depth)', 1, 20, 3)
    min_samples_leaf = st.slider('Mínimo de muestras por hoja (min_samples_leaf)', 1, 30, 10)

    model = train(X_train, y_train, modelSelection, max_depth=max_depth, n_estimators=n_estimators,
                  min_samples_leaf=min_samples_leaf, n_components=n_components)

if featureMode == 'Reducción con PCA':
    st.write('Varianza explicada', round(model.named_steps['pca'].explained_variance_ratio_.sum(), 3))

st.subheader('Datos de la carrera')
year = st.selectbox('Año', sorted(df['Year'].unique()))
location = st.selectbox('Circuito', sorted(df['Location'].unique()))
weather = st.selectbox('Clima', sorted(df['Weather'].unique()))
quali = st.slider('Posición de clasificación', 1, 20, 3)
start = st.slider('Posición de salida', 1, 20, 3)

pred = predict(model, year, location, weather, quali, start)
st.success(f'Posición final predicha: {pred}')

fig, ax = plt.subplots()
ax.bar(['Salida', 'Predicción'], [start, pred])
ax.set_ylabel('Posición')
st.pyplot(fig)

st.subheader('Métricas en prueba')
pred_test = model.predict(X_test)
mae, mse, rmse, r2 = evaluate(y_test, pred_test)
st.write('MAE', round(mae, 3))
st.write('MSE', round(mse, 3))
st.write('RMSE', round(rmse, 3))
st.write('R²', round(r2, 3))

mae_start, mse_start, rmse_start, r2_start = evaluate(y_test, X_test['StartPosition'])
st.write('MAE de la referencia "Usar StartPosition"', round(mae_start, 3))

fig, ax = plt.subplots()
ax.scatter(y_test, pred_test)
ax.plot([1, 20], [1, 20], color='red', linestyle='--')
ax.set_xlabel('Posición real')
ax.set_ylabel('Posición predicha')
st.pyplot(fig)
