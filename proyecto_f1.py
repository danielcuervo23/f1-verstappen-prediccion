# Predicción de la posición final de Max Verstappen - Segundo corte
# Equipo: Iván Moros / Juan Daniel Cuervo
# Versión .py de notebook_proyecto.ipynb (mismos bloques de código, en el mismo orden).
# Las interpretaciones y conclusiones están en el notebook.

# 1. Contexto y continuidad
# 1.1 Contexto y problema

# 1.2 Cambios respecto al primer corte
# 1.3 Resultados esperados

# 1.4 Carga de datos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Carga del archivo CSV en un DataFrame (en Colab se descarga con kagglehub)
df = pd.read_csv('data/f1_enhanced_dataset_for_analysis.csv')

print(df.head())

# Número de filas y columnas del dataset completo (primer corte: 6.000 filas y 17 columnas)
print("Filas y columnas del dataset completo:", df.shape)

# 2. Limpieza y análisis exploratorio
# 2.1 Filtrado del piloto

# Registros de Max Verstappen
df_verstappen = df[df['Driver'] == 'Max Verstappen'].copy()

print("Registros de Max Verstappen:", len(df_verstappen))
print("Carreras distintas:", df_verstappen['RaceID'].nunique())

# Filas por carrera (1 = una sola fila por carrera)
print("Filas por carrera:", df_verstappen.groupby('RaceID').size().unique())

# Pilotos y equipos distintos en el subconjunto (1 = la columna es constante)
print("Pilotos distintos:", df_verstappen['Driver'].nunique())
print("Equipos distintos:", df_verstappen['Team'].nunique(), df_verstappen['Team'].unique())

# Posición final promedio de cada piloto (los 20 pilotos, sin abandonos)
promedio_pilotos = df[df['DNF'] == 0].groupby('Driver')['FinishPosition'].mean().sort_values()

plt.figure(figsize=(12, 6))

sns.barplot(x=promedio_pilotos.index, y=promedio_pilotos.values)

plt.title('Posición final promedio de cada piloto')
plt.xlabel('Piloto')
plt.ylabel('Posición final promedio')
plt.xticks(rotation=90)

plt.show()

# Posición promedio de Verstappen frente al resto de pilotos
print("Posición final promedio de Max Verstappen:", round(promedio_pilotos['Max Verstappen'], 2))
print("Promedio más bajo (mejor):", round(promedio_pilotos.min(), 2), "-", promedio_pilotos.idxmin())
print("Promedio más alto (peor):", round(promedio_pilotos.max(), 2), "-", promedio_pilotos.idxmax())

# 2.2 Limpieza

# 1) Tipos de datos
df_verstappen.info()

# Conversión a número de las columnas que deben ser numéricas
# errors='coerce' deja NaN si algún valor no es un número (así se detectaría un tipo incorrecto)
columnas_enteras = ['Year', 'StartPosition', 'QualifyingPosition', 'FinishPosition', 'DNF', 'Points']

for columna in columnas_enteras:
    convertida = pd.to_numeric(df_verstappen[columna], errors='coerce')
    print(f"{columna}: tipo = {df_verstappen[columna].dtype}, valores no numéricos = {convertida.isnull().sum()}")
    df_verstappen[columna] = convertida

# 2) Formatos inconsistentes en las variables de texto
print("Valores de Weather:", df_verstappen['Weather'].unique())

for columna in ['Location', 'Weather', 'Team']:
    categorias = df_verstappen[columna].nunique()
    categorias_normalizadas = df_verstappen[columna].str.lower().str.strip().nunique()
    con_espacios = (df_verstappen[columna] != df_verstappen[columna].str.strip()).sum()
    print(f"{columna}: categorías = {categorias}, sin mayúsculas ni espacios = {categorias_normalizadas}, "
          f"valores con espacios = {con_espacios}")

# Se quitan los espacios sobrantes (si los valores ya están bien escritos no cambia nada)
for columna in ['Location', 'Weather']:
    df_verstappen[columna] = df_verstappen[columna].str.strip()

# 3) Registros duplicados
print("Registros duplicados:", df_verstappen.duplicated().sum())
print("Carreras repetidas (mismo RaceID):", df_verstappen.duplicated(subset='RaceID').sum())

df_verstappen = df_verstappen.drop_duplicates()
print("Registros después de quitar duplicados:", len(df_verstappen))

# 4) Errores de digitación: mínimos y máximos de las variables numéricas
for columna in ['Year', 'StartPosition', 'QualifyingPosition', 'FinishPosition', 'DNF']:
    print(f"{columna}: mínimo = {df_verstappen[columna].min()}, máximo = {df_verstappen[columna].max()}")

# Rangos válidos: posiciones de 1 a 20, años desde 1950 (primera temporada de F1) y DNF solo 0 o 1
fuera_de_rango = df_verstappen[
    (df_verstappen['StartPosition'] < 1) | (df_verstappen['StartPosition'] > 20) |
    (df_verstappen['QualifyingPosition'] < 1) | (df_verstappen['QualifyingPosition'] > 20) |
    (df_verstappen['FinishPosition'] < 1) | (df_verstappen['FinishPosition'] > 20) |
    (df_verstappen['Year'] < 1950) | (df_verstappen['Year'] > 2024) |
    (~df_verstappen['DNF'].isin([0, 1]))
]

print("Registros con valores fuera de rango:", len(fuera_de_rango))

# 5) Valores nulos por columna
print("Valores nulos en total:", df_verstappen.isnull().sum().sum())

print(df_verstappen.isnull().sum())

# 6) Outliers: diagramas de caja de las posiciones
plt.figure(figsize=(10, 5))

sns.boxplot(data=df_verstappen[['StartPosition', 'QualifyingPosition', 'FinishPosition']])

plt.title('Diagramas de caja de las posiciones de Max Verstappen')
plt.xlabel('Variable')
plt.ylabel('Posición')

plt.show()

# Outliers con el método IQR (rango intercuartílico)
for columna in ['Year', 'StartPosition', 'QualifyingPosition', 'FinishPosition']:
    Q1 = df_verstappen[columna].quantile(0.25)
    Q3 = df_verstappen[columna].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df_verstappen[(df_verstappen[columna] < lower_bound) | (df_verstappen[columna] > upper_bound)]
    print(f"{columna}: límites = [{lower_bound}, {upper_bound}], outliers = {len(outliers)}")

# 7) Abandonos (DNF): posición final de las carreras que Verstappen no terminó
print("Registros con DNF = 1:", (df_verstappen['DNF'] == 1).sum())

print("\nFinishPosition de los registros con DNF = 1:")
print(df_verstappen.loc[df_verstappen['DNF'] == 1, 'FinishPosition'].value_counts().sort_index())

# Se quitan los abandonos: su posición final no es una llegada real
df_limpio = df_verstappen[df_verstappen['DNF'] == 0].copy()

print("Filas eliminadas por DNF:", len(df_verstappen) - len(df_limpio))
print("Filas después de la limpieza:", len(df_limpio))

# 2.3 Variables para el modelado (Data Leakage)

# Relación entre Points y FinishPosition en las carreras terminadas
print("¿Points = 26 - FinishPosition cuando DNF = 0?",
      (df_limpio['Points'] == 26 - df_limpio['FinishPosition']).all())

# Se eliminan los identificadores, las variables con Data Leakage y las columnas constantes
columnas_a_eliminar = [
    'RaceID',              # identificador de la carrera
    'RaceName',            # identificador de la carrera (equivale a RaceID)
    'Points',              # Data Leakage
    'DNF',                 # Data Leakage
    'FastestLapTime_sec',  # Data Leakage
    'PitStopCount',        # Data Leakage
    'Overtakes',           # Data Leakage
    'RaceTime_sec',        # Data Leakage
    'TireStrategy',        # exclusión por precaución
    'Driver',              # constante: solo Max Verstappen
    'Team'                 # constante: siempre Red Bull Racing
]

df_modelo = df_limpio.drop(columns=columnas_a_eliminar)

print("Columnas antes:", df_limpio.shape[1], "- columnas después:", df_modelo.shape[1])
print("Variables para el modelado (incluye la variable objetivo):")
print(df_modelo.columns.tolist())

# 2.4 Análisis univariado

# Distribución de la posición final
plt.figure(figsize=(10, 5))

sns.countplot(data=df_modelo, x='FinishPosition')

plt.title('Distribución de la posición final de Max Verstappen')
plt.xlabel('Posición final')
plt.ylabel('Frecuencia')

plt.show()

# Estadísticas de la posición final
print("Moda:", df_modelo['FinishPosition'].mode()[0])
print("Carreras ganadas (posición 1):", (df_modelo['FinishPosition'] == 1).sum())

print(df_modelo['FinishPosition'].describe().round(2))

# Carreras según la condición climática
plt.figure(figsize=(8, 5))

sns.countplot(data=df_modelo, x='Weather')

plt.title('Carreras según la condición climática')
plt.xlabel('Condición climática')
plt.ylabel('Frecuencia')

plt.show()

# Frecuencias de las variables categóricas
print("Carreras por clima:")
print(df_modelo['Weather'].value_counts())

conteo_circuitos = df_modelo['Location'].value_counts()
print("\nCircuitos distintos:", df_modelo['Location'].nunique())
print("Carreras por circuito: mínimo =", conteo_circuitos.min(), ", máximo =", conteo_circuitos.max())

# 2.5 Análisis bivariado

# Posición de clasificación vs posición final
plt.figure(figsize=(12, 6))

sns.boxplot(data=df_modelo, x='QualifyingPosition', y='FinishPosition')

plt.title('Posición de clasificación vs posición final')
plt.xlabel('Posición de clasificación')
plt.ylabel('Posición final')

plt.show()

# Correlación entre QualifyingPosition y FinishPosition
print(df_modelo[['QualifyingPosition', 'FinishPosition']].corr())

# Posición de salida vs posición final
plt.figure(figsize=(12, 6))

sns.boxplot(data=df_modelo, x='StartPosition', y='FinishPosition')

plt.title('Posición de salida vs posición final')
plt.xlabel('Posición de salida')
plt.ylabel('Posición final')

plt.show()

# Correlación entre StartPosition y FinishPosition
print(df_modelo[['StartPosition', 'FinishPosition']].corr())

# ¿Cuántas posiciones gana o pierde Verstappen respecto a su salida?
diferencia_posiciones = df_modelo['FinishPosition'] - df_modelo['StartPosition']

print("Termina en la misma posición de salida:", (diferencia_posiciones == 0).sum(), "carreras")
print("Diferencia absoluta promedio (salida vs llegada):", round(diferencia_posiciones.abs().mean(), 2))

# Condición climática vs posición final
plt.figure(figsize=(8, 5))

sns.boxplot(data=df_modelo, x='Weather', y='FinishPosition')

plt.title('Condición climática vs posición final')
plt.xlabel('Condición climática')
plt.ylabel('Posición final')

plt.show()

# Posición final promedio según la condición climática
print(df_modelo.groupby('Weather')['FinishPosition'].mean().round(2))

# Posición final promedio por circuito (cada circuito tiene pocas carreras)
promedio_circuito = df_modelo.groupby('Location')['FinishPosition'].mean()

print("Mejor circuito:", promedio_circuito.idxmin(), "-", round(promedio_circuito.min(), 2))
print("Peor circuito:", promedio_circuito.idxmax(), "-", round(promedio_circuito.max(), 2))

# Posición final promedio por año
print(df_modelo.groupby('Year')['FinishPosition'].mean().round(2))

# 2.6 Análisis multivariado

# Matriz de correlación de las variables numéricas (sin RaceID ni DNF, que ya es constante)
correlaciones = df_limpio.select_dtypes(include=np.number).drop(columns=['RaceID', 'DNF']).corr()

plt.figure(figsize=(12, 8))

sns.heatmap(correlaciones, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)

plt.title('Matriz de correlación de variables numéricas')

plt.show()

# Posición final promedio según el clima y el grupo de posición de salida
grupo_salida = pd.cut(df_modelo['StartPosition'], bins=[0, 5, 10, 15, 20], labels=['1-5', '6-10', '11-15', '16-20'])

print(df_modelo.pivot_table(values='FinishPosition', index='Weather', columns=grupo_salida,
                      aggfunc='mean', observed=True).round(2))

# 3. Ingeniería y reducción de variables
# 3.1 Variables y codificación

# Variables predictoras (X) y variable objetivo (y)
X = df_modelo.drop(columns=['FinishPosition'])
y = df_modelo['FinishPosition']

print("Variables predictoras:")
print(X.columns.tolist())

print("\nVariable objetivo:")
print(y.name)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Variables categóricas (se codifican con One-Hot) y numéricas (se dejan igual)
columnas_categoricas = ['Location', 'Weather']
columnas_numericas = ['Year', 'StartPosition', 'QualifyingPosition']

# ColumnTransformer aplica un tratamiento distinto a cada grupo de columnas
# sparse_output=False devuelve una matriz normal, que es la que necesitan StandardScaler y PCA
preprocesamiento = ColumnTransformer(
    transformers=[
        ('categoricas', OneHotEncoder(handle_unknown='ignore', sparse_output=False), columnas_categoricas),
        ('numericas', 'passthrough', columnas_numericas)  # 'passthrough' = sin cambios
    ]
)

print("Preprocesamiento definido.")

# 3.2 División entrenamiento / prueba

from sklearn.model_selection import train_test_split

# 80 % entrenamiento y 20 % prueba; random_state=42 permite repetir la división
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Registros en entrenamiento:", len(X_train))
print("Registros en prueba:", len(X_test))

# Comprobación: como hay una fila por carrera, ninguna carrera queda en los dos conjuntos
carreras_train = df_limpio.loc[X_train.index, 'RaceID']
carreras_test = df_limpio.loc[X_test.index, 'RaceID']
print("Carreras en ambos conjuntos:", len(set(carreras_train) & set(carreras_test)))

# Se ajusta el One-Hot solo con el entrenamiento para ver cuántas columnas genera
X_train_codificado = preprocesamiento.fit_transform(X_train)

print("Columnas antes de codificar:", X_train.shape[1])
print("Columnas después del One-Hot:", X_train_codificado.shape[1])
print("Filas de entrenamiento por columna:", round(X_train_codificado.shape[0] / X_train_codificado.shape[1], 1))

# 3.3 Reducción de variables con PCA

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# PCA es sensible a la escala, por eso primero se estandarizan las columnas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_codificado)

# PCA con todos los componentes para ver cuánta varianza explica cada uno
pca = PCA()
pca.fit(X_train_scaled)

varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)

print("Varianza del primer componente:", round(pca.explained_variance_ratio_[0], 3))
print("Varianza de los dos primeros:", round(varianza_acumulada[1], 3))

# Varianza explicada acumulada según el número de componentes
plt.figure(figsize=(10, 5))

plt.plot(range(1, len(varianza_acumulada) + 1), varianza_acumulada, marker='o')
plt.axhline(0.9, color='red', linestyle='--', label='90 %')

plt.title('Varianza explicada acumulada (PCA)')
plt.xlabel('Número de componentes')
plt.ylabel('Varianza acumulada')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.show()

# Número de componentes para llegar al 90 % de la varianza
n_componentes = int(np.argmax(varianza_acumulada >= 0.90)) + 1

print("Componentes para el 90 % de la varianza:", n_componentes, "de", X_train_scaled.shape[1])
print("Varianza explicada:", round(sum(pca.explained_variance_ratio_[:n_componentes]), 3))
print("Componentes para el 95 % de la varianza:", int(np.argmax(varianza_acumulada >= 0.95)) + 1)

# PCA solo con las dos posiciones (las que tienen correlación de 0.94 entre sí)
X_posiciones = StandardScaler().fit_transform(X_train[['StartPosition', 'QualifyingPosition']])

pca_posiciones = PCA(n_components=2)
pca_posiciones.fit(X_posiciones)

print("Varianza explicada por cada componente:", pca_posiciones.explained_variance_ratio_.round(3))

# 4. Modelado
# 4.1 Modelos candidatos

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Los tres modelos (mismos parámetros del primer corte)
modelo_lineal = LinearRegression()
modelo_arbol = DecisionTreeRegressor(max_depth=5, random_state=42)
modelo_bosque = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

# Diccionario para recorrer los modelos con un ciclo for
modelos = {
    'Regresión Lineal': modelo_lineal,
    'Árbol de Decisión': modelo_arbol,
    'Random Forest': modelo_bosque
}

print("Modelos definidos:", list(modelos.keys()))

# 4.2 Validación cruzada (KFold de 5 partes)

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

# 5 partes con mezcla aleatoria; random_state=42 permite repetir la división
validacion = KFold(n_splits=5, shuffle=True, random_state=42)

resultados_cv = []

for usar_pca in ['No', 'Sí']:
    for nombre, modelo in modelos.items():
        if usar_pca == 'No':
            # Pipeline: primero el One-Hot y luego el modelo
            pipeline = Pipeline([('preprocesamiento', preprocesamiento), ('modelo', modelo)])
        else:
            # Con PCA: One-Hot, escalado, reducción y luego el modelo
            pipeline = Pipeline([
                ('preprocesamiento', preprocesamiento),
                ('escalado', StandardScaler()),
                ('pca', PCA(n_components=n_componentes)),
                ('modelo', modelo)
            ])

        mae = -cross_val_score(pipeline, X_train, y_train, cv=validacion, scoring='neg_mean_absolute_error')
        mse = -cross_val_score(pipeline, X_train, y_train, cv=validacion, scoring='neg_mean_squared_error')
        rmse = -cross_val_score(pipeline, X_train, y_train, cv=validacion, scoring='neg_root_mean_squared_error')
        r2 = cross_val_score(pipeline, X_train, y_train, cv=validacion, scoring='r2')

        # Promedio de las 5 partes
        resultados_cv.append({'Modelo': nombre, 'PCA': usar_pca, 'MAE (CV)': mae.mean(),
                              'MSE (CV)': mse.mean(), 'RMSE (CV)': rmse.mean(), 'R² (CV)': r2.mean()})

tabla_cv = pd.DataFrame(resultados_cv).round(3)
print(tabla_cv)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Las dos referencias simples con las mismas 5 partes (no aprenden nada, solo se calculan)
resultados_ref_cv = []

for nombre in ['Predecir el promedio (entrenamiento)', 'Usar StartPosition']:
    mae, mse, rmse, r2 = [], [], [], []

    for filas_train, filas_val in validacion.split(X_train):
        y_val = y_train.iloc[filas_val]

        if nombre == 'Usar StartPosition':
            prediccion = X_train['StartPosition'].iloc[filas_val]
        else:
            prediccion = np.full(len(y_val), y_train.iloc[filas_train].mean())

        mae.append(mean_absolute_error(y_val, prediccion))
        mse.append(mean_squared_error(y_val, prediccion))
        rmse.append(np.sqrt(mean_squared_error(y_val, prediccion)))
        r2.append(r2_score(y_val, prediccion))

    resultados_ref_cv.append({'Referencia': nombre, 'MAE (CV)': np.mean(mae), 'MSE (CV)': np.mean(mse),
                              'RMSE (CV)': np.mean(rmse), 'R² (CV)': np.mean(r2)})

tabla_ref_cv = pd.DataFrame(resultados_ref_cv).round(3)
print(tabla_ref_cv)

# 4.3 Impacto de PCA

# Impacto de PCA: MAE en validación cruzada con y sin reducción
impacto_pca = tabla_cv.pivot(index='Modelo', columns='PCA', values='MAE (CV)')
impacto_pca['Diferencia (Sí - No)'] = impacto_pca['Sí'] - impacto_pca['No']

print(impacto_pca.round(3))

# 4.4 Ajuste de parámetros (GridSearchCV)

from sklearn.model_selection import GridSearchCV

# Valores a probar (en el primer corte los parámetros se fijaron sin ajustar)
parametros = {
    'Árbol de Decisión': {'modelo__max_depth': [2, 3, 4, 5, 7, 10],
                          'modelo__min_samples_leaf': [1, 5, 10, 20]},
    'Random Forest': {'modelo__n_estimators': [100, 300],
                      'modelo__max_depth': [3, 5, 7],
                      'modelo__min_samples_leaf': [1, 5, 10]}
}

resultados_ajuste = []

for nombre, grilla in parametros.items():
    pipeline = Pipeline([('preprocesamiento', preprocesamiento), ('modelo', modelos[nombre])])

    # GridSearchCV prueba todas las combinaciones con las mismas 5 partes de la validación cruzada
    busqueda = GridSearchCV(pipeline, grilla, cv=validacion, scoring='neg_mean_absolute_error')
    busqueda.fit(X_train, y_train)
    print(nombre, "- mejores parámetros:", busqueda.best_params_)

    # Se agrega al diccionario el modelo con los mejores parámetros
    modelos[nombre + ' (ajustado)'] = busqueda.best_estimator_.named_steps['modelo']

    mae_antes = tabla_cv[(tabla_cv['Modelo'] == nombre) & (tabla_cv['PCA'] == 'No')]['MAE (CV)'].iloc[0]
    resultados_ajuste.append({'Modelo': nombre, 'Mejores parámetros': busqueda.best_params_,
                              'MAE (CV) primer corte': mae_antes, 'MAE (CV) ajustado': -busqueda.best_score_})

tabla_ajuste = pd.DataFrame(resultados_ajuste).round(3)
print(tabla_ajuste)

# Validación cruzada de los modelos ajustados con las cuatro métricas (sin PCA)
for nombre in ['Árbol de Decisión (ajustado)', 'Random Forest (ajustado)']:
    pipeline = Pipeline([('preprocesamiento', preprocesamiento), ('modelo', modelos[nombre])])

    mae = -cross_val_score(pipeline, X_train, y_train, cv=validacion, scoring='neg_mean_absolute_error')
    mse = -cross_val_score(pipeline, X_train, y_train, cv=validacion, scoring='neg_mean_squared_error')
    rmse = -cross_val_score(pipeline, X_train, y_train, cv=validacion, scoring='neg_root_mean_squared_error')
    r2 = cross_val_score(pipeline, X_train, y_train, cv=validacion, scoring='r2')

    resultados_cv.append({'Modelo': nombre, 'PCA': 'No', 'MAE (CV)': mae.mean(),
                          'MSE (CV)': mse.mean(), 'RMSE (CV)': rmse.mean(), 'R² (CV)': r2.mean()})

tabla_cv = pd.DataFrame(resultados_cv).round(3)
print(tabla_cv)

# 4.5 Entrenamiento y evaluación final

resultados = []

# Se evalúan los cinco modelos del diccionario (tres del primer corte y dos ajustados), con y sin PCA
for usar_pca in ['No', 'Sí']:
    for nombre, modelo in modelos.items():
        if usar_pca == 'No':
            pipeline = Pipeline([('preprocesamiento', preprocesamiento), ('modelo', modelo)])
        else:
            pipeline = Pipeline([
                ('preprocesamiento', preprocesamiento),
                ('escalado', StandardScaler()),
                ('pca', PCA(n_components=n_componentes)),
                ('modelo', modelo)
            ])

        # Entrenar (el One-Hot, el escalado, el PCA y el modelo aprenden solo de X_train)
        pipeline.fit(X_train, y_train)

        # Predicciones sobre entrenamiento y prueba
        pred_train = pipeline.predict(X_train)
        pred_test = pipeline.predict(X_test)

        # RMSE = raíz cuadrada del error cuadrático medio
        resultados.append({
            'Modelo': nombre,
            'PCA': usar_pca,
            'MAE train': mean_absolute_error(y_train, pred_train),
            'MAE prueba': mean_absolute_error(y_test, pred_test),
            'MSE train': mean_squared_error(y_train, pred_train),
            'MSE prueba': mean_squared_error(y_test, pred_test),
            'RMSE train': np.sqrt(mean_squared_error(y_train, pred_train)),
            'RMSE prueba': np.sqrt(mean_squared_error(y_test, pred_test)),
            'R² train': r2_score(y_train, pred_train),
            'R² prueba': r2_score(y_test, pred_test)
        })

tabla_resultados = pd.DataFrame(resultados).round(3)
print(tabla_resultados)

# 5. Resultados y despliegue
# 5.1 Referencias simples en prueba

# Predicciones de las dos referencias sobre el conjunto de prueba
referencias = {
    'Predecir el promedio (entrenamiento)': np.full(len(y_test), y_train.mean()),
    'Usar StartPosition': X_test['StartPosition']
}

resultados_ref = []

for nombre, prediccion in referencias.items():
    resultados_ref.append({
        'Referencia': nombre,
        'MAE prueba': mean_absolute_error(y_test, prediccion),
        'MSE prueba': mean_squared_error(y_test, prediccion),
        'RMSE prueba': np.sqrt(mean_squared_error(y_test, prediccion)),
        'R² prueba': r2_score(y_test, prediccion)
    })

tabla_referencias = pd.DataFrame(resultados_ref).round(3)
print(tabla_referencias)

# 5.2 Detección de overfitting y underfitting

# Diferencia entre entrenamiento y prueba (valores grandes indicarían overfitting)
diferencias = pd.DataFrame({
    'Modelo': tabla_resultados['Modelo'],
    'PCA': tabla_resultados['PCA'],
    'MAE prueba - MAE train': tabla_resultados['MAE prueba'] - tabla_resultados['MAE train'],
    'R² train - R² prueba': tabla_resultados['R² train'] - tabla_resultados['R² prueba']
}).round(3)

print(diferencias)

# 5.3 Impacto de PCA en prueba
# 5.4 Selección del mejor modelo

# Modelo con mejor resultado en validación cruzada según cada métrica
print("Menor MAE en validación cruzada:", tabla_cv.loc[tabla_cv['MAE (CV)'].idxmin(), ['Modelo', 'PCA']].tolist())
print("Menor RMSE en validación cruzada:", tabla_cv.loc[tabla_cv['RMSE (CV)'].idxmin(), ['Modelo', 'PCA']].tolist())
print("Mayor R² en validación cruzada:", tabla_cv.loc[tabla_cv['R² (CV)'].idxmax(), ['Modelo', 'PCA']].tolist())

# Comparación con la referencia "Usar StartPosition" (mismas 5 partes)
mae_start_cv = tabla_ref_cv.loc[tabla_ref_cv['Referencia'] == 'Usar StartPosition', 'MAE (CV)'].iloc[0]
print("\nMenor MAE (CV) de un modelo:", tabla_cv['MAE (CV)'].min())
print("MAE (CV) de Usar StartPosition:", mae_start_cv)

# MAE en validación cruzada de cada modelo, con y sin PCA
plt.figure(figsize=(12, 6))

sns.barplot(data=tabla_cv, x='Modelo', y='MAE (CV)', hue='PCA')
plt.axhline(mae_start_cv, color='red', linestyle='--', label='Usar StartPosition')

plt.title('MAE en validación cruzada por modelo (más bajo es mejor)')
plt.xlabel('Modelo')
plt.ylabel('MAE (posiciones)')
plt.xticks(rotation=15)
plt.legend(title='PCA')

plt.show()

# Posición real vs posición predicha en prueba con el mejor modelo por CV (no usa PCA)
nombre_mejor = tabla_cv.loc[tabla_cv['MAE (CV)'].idxmin(), 'Modelo']
pipeline_final = Pipeline([('preprocesamiento', preprocesamiento), ('modelo', modelos[nombre_mejor])])
pipeline_final.fit(X_train, y_train)
pred_final = pipeline_final.predict(X_test)

plt.figure(figsize=(8, 5))

sns.scatterplot(x=y_test, y=pred_final)
plt.plot([1, 20], [1, 20], color='red', linestyle='--', label='Predicción perfecta')

plt.title('Posición real vs posición predicha en prueba - ' + nombre_mejor)
plt.xlabel('Posición real')
plt.ylabel('Posición predicha')
plt.legend()

plt.show()

# 5.5 Despliegue en Streamlit

# 6. Conclusiones

# Respuesta a la pregunta del proyecto (las conclusiones completas están en el notebook)
print("Mejor modelo por validación cruzada:", nombre_mejor)
print("MAE (CV) del mejor modelo:", tabla_cv['MAE (CV)'].min())
print("MAE (CV) de Usar StartPosition:", mae_start_cv)
print("¿Algún modelo supera a Usar StartPosition en validación cruzada?", tabla_cv['MAE (CV)'].min() < mae_start_cv)

# 7. Limitaciones y trabajo futuro

# 8. Declaración de uso de IA
