import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data():
    df = pd.read_csv('data/f1_enhanced_dataset_for_analysis.csv')
    df = df[df['Driver'] == 'Max Verstappen']
    return df

def preprocessing(df):
    # Se quitan los abandonos (DNF): su posición final no es una llegada real
    df = df[df['DNF'] == 0]
    X = df[['Year', 'Location', 'Weather', 'StartPosition', 'QualifyingPosition']]
    y = df['FinishPosition']
    return X, y

def train(X_train, y_train, modelSelection, max_depth=5, n_estimators=100, min_samples_leaf=1, n_components=None):
    preprocesamiento = ColumnTransformer(
        transformers=[
            ('categoricas', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Location', 'Weather']),
            ('numericas', 'passthrough', ['Year', 'StartPosition', 'QualifyingPosition'])
        ]
    )

    if modelSelection == 'Regresión Lineal':
        modelo = LinearRegression()
    elif modelSelection == 'Árbol de Decisión':
        modelo = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
    elif modelSelection == 'Random Forest':
        modelo = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)

    if n_components is None:
        model = Pipeline([('preprocesamiento', preprocesamiento), ('modelo', modelo)])
    else:
        model = Pipeline([
            ('preprocesamiento', preprocesamiento),
            ('escalado', StandardScaler()),
            ('pca', PCA(n_components=n_components)),
            ('modelo', modelo)
        ])

    model.fit(X_train, y_train)
    return model

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

def predict(model, year, location, weather, quali, start):
    row = pd.DataFrame({
        'Year': [year],
        'Location': [location],
        'Weather': [weather],
        'StartPosition': [start],
        'QualifyingPosition': [quali]
    })
    pred = round(model.predict(row)[0])
    return int(np.clip(pred, 1, 20))
