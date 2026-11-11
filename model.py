import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data():
  df = pd.read_csv('data/f1_enhanced_dataset_for_analysis.csv')
  return df[df['Driver'] == 'Max Verstappen']

def preprocessing(df):
  # quitamos los abandonos (DNF) porque no son una posicion real de llegada
  df = df[df['DNF'] != 1]
  # solo datos que se saben antes de la carrera
  X = df[['Year', 'Location', 'Weather', 'StartPosition', 'QualifyingPosition']]
  y = df['FinishPosition']
  X = pd.get_dummies(X, columns=['Location', 'Weather'])
  return X, y

def train(X, y, name):
  if name == 'Regresión Lineal':
    model = LinearRegression()
  elif name == 'Árbol de Decisión':
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
  else:
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
  model.fit(X, y)
  return model

def evaluate(y_true, y_pred):
  mae = mean_absolute_error(y_true, y_pred)
  mse = mean_squared_error(y_true, y_pred)
  rmse = np.sqrt(mse)
  r2 = r2_score(y_true, y_pred)
  return mae, mse, rmse, r2

def predict(model, X_columns, year, location, weather, quali, start):
  row = pd.DataFrame({'Year': [year], 'Location': [location], 'Weather': [weather],
                      'StartPosition': [start], 'QualifyingPosition': [quali]})
  row = pd.get_dummies(row, columns=['Location', 'Weather'])
  # un circuito nuevo no esta en el entrenamiento, queda en 0
  row = row.reindex(columns=X_columns, fill_value=0)
  pred = round(model.predict(row)[0])
  return int(np.clip(pred, 1, 20))
