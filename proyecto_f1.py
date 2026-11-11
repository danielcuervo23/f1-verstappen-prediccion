import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from model import load_data, preprocessing, train, evaluate

# 1. Cargar datos
df_all = pd.read_csv('data/f1_enhanced_dataset_for_analysis.csv')
print('Dataset completo:', df_all.shape)
df = load_data()
print('Filas de Verstappen:', df.shape)

# 2. Limpieza
print('Valores nulos:', df.isnull().sum().sum())
print('Duplicados:', df.duplicated().sum())
print('Filas con DNF (abandonos) que se quitan:', (df['DNF'] == 1).sum())
# Se usan solo 5 columnas: las demás (Points, RaceTime_sec, etc.) se saben hasta después de la carrera
X, y = preprocessing(df)
print('Forma final de X:', X.shape)

# 3. Exploración
clean = df[df['DNF'] != 1]
print(clean.describe())

plt.hist(clean['FinishPosition'], bins=range(1, 22), align='left')
plt.title('Posición final de Verstappen')
plt.xlabel('Posición')
plt.ylabel('Carreras')
plt.show()

plt.scatter(clean['StartPosition'], clean['FinishPosition'])
plt.title('Salida vs llegada')
plt.xlabel('Posición de salida')
plt.ylabel('Posición final')
plt.show()

clean.boxplot(column='FinishPosition', by='Weather')
plt.show()

corr = clean.select_dtypes(include=['int64', 'float64']).corr()
plt.imshow(corr, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title('Correlación')
plt.show()

print('Corr StartPosition:', clean['StartPosition'].corr(clean['FinishPosition']))
print('Corr QualifyingPosition:', clean['QualifyingPosition'].corr(clean['FinishPosition']))

mean_pos = df_all.groupby('Driver')['FinishPosition'].mean().sort_values()
mean_pos.plot(kind='bar')
plt.title('Posición promedio de cada piloto')
plt.ylabel('Posición promedio')
plt.show()

# 4. PCA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pca = PCA().fit(StandardScaler().fit_transform(X_train))
cum_var = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o')
plt.axhline(0.9, linestyle='--')
plt.title('Varianza acumulada')
plt.xlabel('Componentes')
plt.show()

n = int(np.argmax(cum_var >= 0.9)) + 1
print('Componentes para 90% de varianza:', n, 'de', X.shape[1])
print('Varianza explicada:', cum_var[n - 1])

# 5. Modelos
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for use_pca in [False, True]:
  for name in ['Regresión Lineal', 'Árbol de Decisión', 'Random Forest']:
    model = train(X_train, y_train, name)
    if use_pca:
      model = make_pipeline(StandardScaler(), PCA(n), clone(model)).fit(X_train, y_train)
    cv = -cross_val_score(clone(model), X_train, y_train, cv=kf,
                          scoring='neg_mean_absolute_error').mean()
    train_mae = evaluate(y_train, model.predict(X_train))[0]
    mae, mse, rmse, r2 = evaluate(y_test, model.predict(X_test))
    rows.append({'modelo': name, 'pca': 'si' if use_pca else 'no', 'cv_MAE': cv,
                 'train_MAE': train_mae, 'test_MAE': mae, 'test_MSE': mse,
                 'test_RMSE': rmse, 'test_R2': r2})

# Reglas simples: terminar donde salió y siempre el promedio
pred_train = X_train['StartPosition']
pred_test = X_test['StartPosition']
mae, mse, rmse, r2 = evaluate(y_test, pred_test)
simple_mae = evaluate(y_train, pred_train)[0]
rows.append({'modelo': 'termina donde salió', 'pca': '-', 'cv_MAE': simple_mae,
             'train_MAE': simple_mae, 'test_MAE': mae, 'test_MSE': mse,
             'test_RMSE': rmse, 'test_R2': r2})
mean_train = np.full(len(y_train), y_train.mean())
mean_test = np.full(len(y_test), y_train.mean())
mae, mse, rmse, r2 = evaluate(y_test, mean_test)
mean_mae = evaluate(y_train, mean_train)[0]
rows.append({'modelo': 'siempre el promedio', 'pca': '-', 'cv_MAE': mean_mae,
             'train_MAE': mean_mae, 'test_MAE': mae, 'test_MSE': mse,
             'test_RMSE': rmse, 'test_R2': r2})
results = pd.DataFrame(rows)

# 6. Resultados
print(results.round(2).to_string(index=False))
models = results[results['pca'] != '-']
best = models.loc[models['cv_MAE'].idxmin()]
print('Mejor modelo por cv_MAE:', best['modelo'], '(PCA:', best['pca'] + ')', round(best['cv_MAE'], 2))
print('MAE de la regla simple (termina donde salió):', round(simple_mae, 2))

labels = models['modelo'] + ' PCA ' + models['pca']
plt.bar(labels, models['cv_MAE'])
plt.axhline(simple_mae, color='red', linestyle='--', label='termina donde salió')
plt.xticks(rotation=45, ha='right')
plt.ylabel('MAE en validación cruzada')
plt.legend()
plt.tight_layout()
plt.show()

# Conclusión:
# - En validación cruzada ningún modelo le gana a "termina donde salió".
# - Usar PCA empeoró a todos los modelos.
# - Los árboles sobreajustan: su error de entrenamiento es mucho menor que el de validación cruzada.
# - Los datos parecen sintéticos (Verstappen queda casi en el promedio de todos los pilotos).
