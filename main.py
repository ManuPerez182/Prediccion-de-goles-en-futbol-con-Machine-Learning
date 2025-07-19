import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

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