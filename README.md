# Predicción de la posición final de Max Verstappen

Equipo: Iván Moros / Juan Daniel Cuervo

En este proyecto queremos responder una pregunta: ¿en qué posición va a terminar
Max Verstappen una carrera de F1 si solo usamos lo que se sabe antes de la carrera
(año, circuito, clima, posición de clasificación y posición de salida)?
Es un problema de regresión. Probamos regresión lineal, árbol de decisión y
random forest (con y sin PCA) y los comparamos contra una regla simple:
"termina donde salió". Los datos son de Kaggle (Formula 1 race dataset for predictive analysis).

## Cómo correrlo

1. `pip install -r requirements.txt`
2. `python proyecto_f1.py` (análisis completo)
3. `streamlit run app.py` (app)

Archivos: `model.py` (funciones), `proyecto_f1.py` (análisis), `app.py` (app) y `notebook_proyecto.ipynb`.

Enlace de la app: (pendiente)

Notebook en Colab: (pendiente)
