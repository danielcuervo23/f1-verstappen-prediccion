# Predicción de la posición final de Max Verstappen

Proyecto de Inteligencia Artificial (segundo corte) - UNAB

Equipo: Iván Moros / Juan Daniel Cuervo

## Descripción

En el primer corte planteamos estimar la posición final de un piloto de F1 usando solo información que se conoce antes de la carrera. En este segundo corte mantenemos el mismo dataset y la misma pregunta, pero enfocada en un piloto: ¿en qué posición va a terminar Max Verstappen una carrera si solo sabemos el año, el circuito, el clima pronosticado, su posición de clasificación y su posición de salida?

Es un problema de aprendizaje supervisado (regresión). Limpiamos los datos (quitamos los 20 abandonos: 300 → 280 carreras), hicimos el análisis exploratorio, probamos PCA y comparamos Regresión Lineal, Árbol de Decisión y Random Forest con validación cruzada (KFold de 5 partes) y las métricas MAE, MSE, RMSE y R².

Resultado principal: el mejor modelo en validación cruzada es un Random Forest ajustado (`max_depth=3`, `min_samples_leaf=10`) con MAE de 3.84 posiciones, pero no le gana a la regla simple "Usar StartPosition" (predecir que termina donde salió), que tiene MAE de 3.57. PCA empeoró a los tres modelos. Todo el análisis y las conclusiones están en el notebook.

Datos: [Formula 1 Race Dataset for Predictive Analysis](https://www.kaggle.com/datasets/usman136/formula-1-race-dataset-for-predictive-analysis) (Kaggle).

## Archivos

- `notebook_proyecto.ipynb`: notebook de Colab con todo el análisis, las tablas y las interpretaciones.
- `proyecto_f1.py`: el mismo análisis del notebook en un archivo .py.
- `app.py`: la app de Streamlit (interfaz).
- `model.py`: las funciones que usa la app (cargar datos, preprocesar, entrenar, evaluar y predecir).
- `requirements.txt`: librerías necesarias.
- `data/f1_enhanced_dataset_for_analysis.csv`: el dataset.
- `IAProyecto.pptx`: la presentación del primer corte.

## Estructura de la app

Seguimos la estructura vista en clase (`app.py`, `model.py`, `requirements.txt`):

- `app.py` es el frontend: muestra la vista previa del dataset, deja elegir el modo de variables (selección manual o PCA), el modelo y sus parámetros (`max_depth`, `min_samples_leaf`, `n_estimators`) y los datos de la carrera.
- `model.py` es el backend: `load_data()`, `preprocessing(df)`, `train(...)`, `evaluate(...)` y `predict(...)`. El modelo se arma con el mismo `Pipeline` del notebook (`ColumnTransformer` + `OneHotEncoder(handle_unknown='ignore')` + el modelo) y se entrena en vivo al abrir la app con la misma división 80/20 (`random_state=42`).
- La app muestra la posición final predicha, una gráfica de salida vs predicción, las métricas en prueba (MAE, MSE, RMSE y R²), la comparación con "Usar StartPosition" y una gráfica de posición real vs predicha.

## Cómo correrlo

1. `pip install -r requirements.txt`
2. `python proyecto_f1.py` (análisis completo)
3. `streamlit run app.py` (app)

El notebook también se puede abrir directo en Google Colab: descarga los datos de Kaggle con `kagglehub`.

## Enlaces

- App en Streamlit: (pendiente)
- Notebook en Colab: (pendiente)

## Uso de IA

Usamos Claude Code (Anthropic) como apoyo para planear, escribir y revisar código y adaptarlo al estilo de la clase. La declaración completa (herramienta, versión del modelo, para qué se usó y nuestros aportes) está en la sección 8 del notebook.
