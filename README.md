# ⚽ Predicción de Goles en Partidos de Fútbol (FIFA World Cup)

Este proyecto consiste en desarrollar un modelo de Machine Learning capaz de **predecir la cantidad de goles del equipo local** en un partido de fútbol, utilizando estadísticas reales de encuentros anteriores de la Copa Mundial de la FIFA.

Se emplea un enfoque de **aprendizaje supervisado con regresión**, entrenando al modelo con información numérica del partido (posesión, remates, tarjetas, etc.) para que aprenda a estimar cuántos goles podría marcar el equipo local en futuros partidos similares.

## Dataset

- **Fuente**: [FIFA World Cup Match Stats - Kaggle](https://www.kaggle.com/datasets/abecklas/fifa-world-cup)
- **Descripción**: Contiene estadísticas detalladas de cientos de partidos internacionales, incluyendo posesión, remates, tarjetas, faltas, etc.
- **Uso**: Se procesó y filtró para conservar únicamente columnas numéricas relevantes para la predicción de goles.

## Tecnologías utilizadas

| Herramienta        | Rol en el proyecto                                                   |
|--------------------|----------------------------------------------------------------------|
| **Python**         | Lenguaje principal del desarrollo                                    |
| **pandas**         | Carga, limpieza y análisis de datos en formato tabular               |
| **scikit-learn**   | Separación de datos, entrenamiento del modelo, evaluación de métricas|
| **TensorFlow**     | Entrenamiento de modelos más complejos (redes neuronales)            |
| **matplotlib**     | Visualización de resultados y métricas                               |

## Metodología

- **Regresión supervisada**: el modelo predice un valor numérico continuo (goles).
- **Normalización de datos**: para escalar las variables y mejorar el rendimiento del modelo.
- **Red neuronal básica**: construida con TensorFlow y entrenada para ajustar los datos.
- **Métricas de evaluación**:
  - **MAE (Mean Absolute Error)**: error promedio entre los valores reales y los predichos.
  - **RMSE (Root Mean Squared Error)**: penaliza errores grandes, útil para evaluar precisión.

---

Este proyecto busca no solo aplicar técnicas de ML sobre datos deportivos reales, sino también ser una prueba práctica de conocimientos en programación, análisis de datos y aprendizaje automático.

# Machine Learning

**Machine Learning (ML)** es una rama de la inteligencia artificial que permite a las computadoras aprender de los datos y hacer predicciones o tomar decisiones sin estar programadas explícitamente para cada caso.

Existen varios tipos de aprendizaje automático, entre los más comunes están:

- **Aprendizaje supervisado**: El modelo aprende a partir de un conjunto de datos etiquetado, es decir, cada ejemplo tiene una entrada (features) y una salida (target) conocida. Es ideal para tareas como predicción de precios, clasificación de imágenes o, en este caso, predicción de goles.

- **Aprendizaje no supervisado**: No hay etiquetas. El modelo intenta encontrar estructuras ocultas en los datos. Se usa en agrupamientos (clustering) o reducción de dimensionalidad.

- **Aprendizaje por refuerzo**: El modelo aprende tomando decisiones en un entorno y recibe recompensas o penalizaciones en función de su comportamiento. Se utiliza mucho en juegos, robótica y sistemas de recomendación.

### ¿Qué tipo de aprendizaje usamos en este proyecto?

Este proyecto utiliza **aprendizaje supervisado** con un modelo de **regresión**, ya que buscamos predecir un valor numérico continuo: la cantidad de goles que hará el equipo local en un partido de fútbol, a partir de estadísticas del encuentro.

---
# Setup del entorno de desarrollo

Este proyecto fue desarrollado en macOS utilizando Python y librerías de Machine Learning. A continuación se detalla el proceso completo de instalación, configuración del entorno virtual y setup del repositorio en GitHub.

## Instalación de Python

Primero instalé Python utilizando [Homebrew](https://brew.sh/), el gestor de paquetes para macOS:

```bash
brew install python
```
Luego verifiqué que la instalación fue exitosa:

```bash
python3 --version
pip3 --version
```

## Entorno virtual

Un entorno virtual es un espacio aislado donde se instalan las librerías de Python para un solo proyecto.
Esto evita conflictos entre proyectos distintos que usan diferentes versiones de las mismas librerías.

- Desde la raíz del proyecto:
```bash
python3 -m venv futbol-env
```
- Luego activé el entorno con:
```bash
source futbol-env/bin/activate
```
Nota: Cada vez que vuelvo a trabajar en el proyecto, debo reactivar el entorno con el mismo comando.

## Instalación de librerías
Con el entorno virtual activado, instalé las librerías necesarias para trabajar con análisis de datos, visualización, Machine Learning y redes neuronales:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```
Luego guardé esas dependencias en un archivo para que cualquier persona pueda instalarlas fácilmente:
```bash
pip freeze > requirements.txt
```
Si qusiera instalar todo en otra máquina con el mismo entorno virtual:
```bash
pip install -r requirements.txt
```
## Configuración de Git y GitHub
Inicialicé el repositorio Git dentro del proyecto:
```bash
git init
```
Luego conecté mi repositorio local con uno ya creado en GitHub:
```bash
git remote add origin https://github.com/ManuPerez182/Prediccion-de-goles-en-futbol-con-Machine-Learning.git
```
Agregué y subí los primeros archivos al repositorio remoto:
```bash
git add .
git commit -m "Primer commit con estructura del proyecto"
git push -u origin main
```
## Archivo .gitignore
Para evitar subir archivos innecesarios o generados automáticamente, creé un archivo .gitignore con el siguiente contenido:
```bash
# Entorno virtual
futbol-env/

# Archivos temporales y compilados
__pycache__/
*.py[cod]
*.log
.vscode/
.DS_Store

# Modelos y datasets locales
*.h5
Dataset/
```
---
# Preprocesamiento y preparación de los datos

En esta sección realizamos todos los pasos necesarios para dejar el dataset listo para ser utilizado por nuestro modelo de Machine Learning. Estos pasos incluyen:

- Cargar y explorar los datos
- Seleccionar variables relevantes
- Limpiar datos innecesarios o incompletos
- Separar variables predictoras (features) y objetivo (target)
- Dividir en conjuntos de entrenamiento y prueba

Todo esto forma parte de lo que se conoce como pipeline de preprocesamiento.

Un **pipeline de preprocesamiento** es la secuencia ordenada de pasos que aplicamos a los datos antes de entrenar un modelo de Machine Learning.

El objetivo es preparar los datos de forma estandarizada y reproducible, asegurando que el modelo trabaje con información limpia, consistente y adecuada.

## Carga y exploración inicial del dataset

En este primer paso vamos a cargar el dataset `match_stats.csv`, que contiene estadísticas de partidos de fútbol internacional. El archivo está ubicado dentro de la carpeta `Dataset/`.

Para esta tarea utilizamos la librería **pandas**.

####  ¿Qué es pandas?

`pandas` es una librería de Python especializada en el manejo y análisis de datos estructurados, como tablas. Su estructura principal se llama **DataFrame**, y permite trabajar con los datos de forma similar a una planilla de cálculo (como Excel), pero usando código.

Es una herramienta esencial en cualquier proyecto de Machine Learning porque nos permite:

- Cargar datasets (CSV, Excel, SQL, etc.)
- Filtrar y transformar columnas
- Eliminar datos vacíos
- Realizar análisis y limpieza previa al entrenamiento del modelo

#### Código

```python
import pandas as pd

# Cargar el archivo CSV dentro de la carpeta Dataset
df = pd.read_csv("Dataset/match_stats.csv")

# Mostrar las primeras filas del dataset
print(df.head())

# Ver información general de las columnas y tipos de datos
print(df.info())
```
## Selección y limpieza de datos

Una vez que cargamos el dataset, es importante preparar los datos correctamente para el modelo de Machine Learning. Esto implica dos pasos principales: seleccionar solo las columnas útiles y limpiar los datos si es necesario.

####  ¿Por qué seleccionamos solo columnas numéricas?

En Machine Learning, los modelos trabajan exclusivamente con datos numéricos. Por eso, descartamos columnas con texto (como los nombres de los equipos) o información que no aporta directamente a la predicción que queremos hacer.

Nos quedamos con columnas que:
- Contienen datos **cuantificables** (por ejemplo: posesión, remates, faltas).
- Están relacionadas al **rendimiento del equipo local y visitante**.
- Están listas para usarse sin procesamiento adicional.

Esta técnica se llama **Feature Selection**

Es una parte fundamental del **preprocesamiento de datos** en Machine Learning. Tiene como objetivo:

- Eliminar ruido o datos irrelevantes
- Reducir la complejidad del modelo (menos columnas = menos esfuerzo computacional)
- Mejorar la calidad del entrenamiento
- Evitar el sobreajuste (*overfitting*)

## Selección de columnas

Como queremos predecir la cantidad de goles del equipo local (`hgoals`), seleccionamos únicamente las columnas **numéricas** relacionadas al rendimiento del equipo local.

#### Código

```python
# Selección de columnas numéricas útiles
df_model = df[[
    'hgoals',         # Target: goles del equipo local
    'hPossesion',
    'hshots',
    'hshotsOnTarget',
    'hfouls',
    'hyellowCards',
    'hredCards',
    'hsaves'
]]
# Filtrar el DataFrame para quedarnos solo con las columnas relevantes
df = df[columns]
```
## Separación en features y target

En Machine Learning supervisado, el objetivo es que el modelo **aprenda a hacer predicciones basándose en ejemplos anteriores**. Para lograr esto, es necesario separar los datos en dos componentes fundamentales:

- **Features (`X`)**: Son las variables de entrada, también conocidas como características. Contienen la información que el modelo va a analizar para intentar encontrar patrones. En nuestro caso, estas variables incluyen estadísticas como la posesión, remates al arco, tarjetas y otras métricas del rendimiento de los equipos.

- **Target (`y`)**: Es la variable de salida u objetivo. Representa el valor que queremos que el modelo aprenda a predecir. En este proyecto, la variable objetivo es `hgoals`, es decir, la cantidad de goles que hace el equipo local.

Separar estos dos componentes permite al modelo construir una **relación matemática o estadística entre los inputs (`X`) y el output (`y`)**. Durante el entrenamiento, el modelo analiza cómo cambian los valores de `y` en función de las distintas combinaciones de valores en `X`. Así, cuando reciba nuevos datos similares, podrá estimar cuál sería el valor de salida más probable.

Este paso es clave para todo el proceso de entrenamiento y evaluación, y se aplica en la mayoría de los modelos de regresión, clasificación y predicción.

#### Código

```python
# 'hgoals' es la variable target
y = df['hgoals']

# Todas las demás columnas son features
X = df.drop('hgoals', axis=1)

# Verificamos las dimensiones
print("Tamaño de X:", X.shape)
print("Tamaño de y:", y.shape)
```
## Separación en conjunto de entrenamiento y prueba

Antes de entrenar nuestro modelo, es fundamental dividir el dataset en dos partes:

- **Conjunto de entrenamiento** (`train`): con el que el modelo aprenderá.
- **Conjunto de prueba** (`test`): con el que evaluaremos qué tan bien predice el modelo con datos que nunca vio.

Esta práctica es clave en el **aprendizaje supervisado**, ya que permite comprobar si el modelo realmente generaliza o simplemente memorizó los datos.

Dividir el dataset nos ayuda a:

- Detectar sobreajuste (cuando el modelo aprende demasiado bien los datos de entrenamiento y falla en datos nuevos).
- Obtener una métrica objetiva de rendimiento sobre datos "no vistos".

Una división habitual es **80% para entrenamiento y 20% para prueba**.

### ¿Qué es `scikit-learn`?

[`scikit-learn`](https://scikit-learn.org/) es una de las librerías más utilizadas para Machine Learning en Python. Nos proporciona herramientas para:

- Entrenar modelos de regresión y clasificación
- Preprocesar y transformar datos
- Evaluar modelos con métricas estándar
- Automatizar flujos de trabajo con pipelines

---

### ¿Qué hace `train_test_split`?

Es una función que permite dividir fácilmente los datos en conjunto de entrenamiento y prueba. Pertenece al módulo `sklearn.model_selection`.

- `X`: representa las features del partido (posesión, remates, faltas, etc.).

- `y`: representa el target (cantidad de goles del equipo local).

- `test_size=0.2`: 20% del dataset se destina a prueba.

- `random_state=42`: asegura que la división siempre sea la misma si se vuelve a ejecutar.

**El resultado son 4 conjuntos:**

- `X_train`, `y_train`: para entrenar el modelo

- `X_test`, `y_test`: para probar el modelo

#### Código

```python
# División en 80% entrenamiento y 20% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
