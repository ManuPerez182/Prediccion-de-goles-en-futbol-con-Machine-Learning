import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Cargar dataset
df = pd.read_csv('Dataset/match_stats.csv')

# Ver primeras filas
print(df.head())

# Ver estructura
print(df.info())

# Columnas que vamos a usar
columns = [
    'hgoals',               # Target (lo que queremos predecir)
    'hPossesion', 'aPossesion',
    'hshotsOnTarget', 'ashotsOnTarget',
    'hyellowCards', 'ayellowCards',
    'hredCards', 'aredCards',
    'hfouls', 'afouls',
    'hsaves', 'asaves'
]

# Filtramos el DataFrame original para quedarnos solo con esas columnas
df = df[columns]

# Verificamos si hay valores faltantes en las columnas seleccionadas
if df.isnull().sum().sum() > 0:
    print("Se encontraron valores nulos. Eliminando filas incompletas...")
    # Si hay valores nulos, los eliminamos
    df = df.dropna()
else:
    print("No se encontraron valores nulos.")

# 'hgoals' es la variable objetivo (target)
y = df['hgoals']

# Todas las demás columnas son features (predictoras)
X = df.drop('hgoals', axis=1)

# Verificamos las dimensiones
print("Tamaño de X:", X.shape)
print("Tamaño de y:", y.shape)

# Separar en conjunto de entrenamiento y prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear el objeto scaler
scaler = StandardScaler()

# Ajustar y transformar los datos de entrenamiento
X_train_scaled = scaler.fit_transform(X_train)

# Usar el mismo scaler para transformar el test
X_test_scaled = scaler.transform(X_test)

# Crear el modelo
model = LinearRegression()

# Entrenarlo con los datos normalizados
model.fit(X_train_scaled, y_train)

# Hacer predicciones con el conjunto de prueba
y_pred = model.predict(X_test_scaled)

# Error Absoluto Medio (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Error Cuadrático Medio (MSE) y Raíz (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Coeficiente de Determinación (R^2)
r2 = r2_score(y_test, y_pred)

# Mostrar resultados
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

# Guardar el modelo y el scaler en la carpeta 'models/'
joblib.dump(model, 'models/modelo_goles.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("✅ Modelo y scaler guardados en la carpeta 'models/' correctamente.")

# Crear la figura

# Calculamos el error absoluto
errores = np.abs(y_test - y_pred)

# Clasificamos los puntos por nivel de error
colores = []
for e in errores:
    if e == 0:
        colores.append("green")  # Predicción exacta
    elif e <= 1:
        colores.append("blue")   # Error moderado (cerca)
    else:
        colores.append("red")    # Error grande (lejos)

# Inicializamos contadores
total = len(errores)
verde = np.sum(errores == 0)
azul = np.sum((errores > 0) & (errores <= 1))
rojo = np.sum(errores > 1)

# Calculamos los porcentajes
p_verde = (verde / total) * 100
p_azul = (azul / total) * 100
p_rojo = (rojo / total) * 100

# Mostramos resultados
print(f"Predicción exacta: {p_verde:.2f}%")
print(f"Error moderado (≤ 1 gol): {p_azul:.2f}%")
print(f"Error grande (> 1 gol): {p_rojo:.2f}%")

# Gráfico

# Tamaño de la figura
plt.figure(figsize=(8, 6))

# Dibuja los puntos: cada punto representa un partido
plt.scatter(y_test, y_pred, c=colores, alpha=0.6)

# Dibuja la línea y = x (ideal)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='y = x')

# Etiquetas y formato
plt.xlabel('Goles reales')
plt.ylabel('Goles predichos')
plt.title('Predicción vs Realidad')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()