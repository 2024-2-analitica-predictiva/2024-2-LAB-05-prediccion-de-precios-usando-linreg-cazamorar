#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#



import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import gzip
import pickle
import json

# Paso 1: Cargar y preprocesar los datos
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Age'] = 2021 - df['Year']
    df = df.drop(columns=['Year', 'Car_Name'])
    return df

train_data = preprocess_data('files/input/train_data.csv')
test_data = preprocess_data('files/input/test_data.csv')

# Paso 2: Dividir los datos en x e y
x_train = train_data.drop(columns=['Present_Price'])
y_train = train_data['Present_Price']

x_test = test_data.drop(columns=['Present_Price'])
y_test = test_data['Present_Price']




# Paso 3: Crear el pipeline
numeric_features = ['Selling_Price', 'Driven_kms', 'Age']
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission', 'Owner']

preprocessor = ColumnTransformer([
    ('num', MinMaxScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=f_regression)),
    ('regressor', LinearRegression())
])

# Paso 4: Optimizar hiperparámetros
param_grid = {
    'feature_selection__k': [5, 10, 13, 15, 'all'], 
    'regressor__fit_intercept': [True, False],
    'regressor__copy_X': [True, False],           # Copiar X (útil en pipelines complejos)
    'regressor__positive': [True, False]          # Forzar coeficientes positivos
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring='neg_mean_absolute_error',
    n_jobs=-1, 
    refit=True,
    verbose=1
)
grid_search.fit(x_train, y_train)

# Imprimir los mejores parámetros encontrados
print("Mejores parámetros encontrados:", grid_search.best_params_)

# Paso 5: Guardar el modelo comprimido
model_path = "files/models/model.pkl.gz"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with gzip.open(model_path, 'wb') as f:
    pickle.dump(grid_search, f)

# Paso 6: Calcular métricas y guardar resultados
metrics = []
for dataset, x, y in [('train', x_train, y_train), ('test', x_test, y_test)]:
    predictions = grid_search.predict(x)
    r2 = r2_score(y, predictions)
    mse = mean_squared_error(y, predictions)
    mad = mean_absolute_error(y, predictions)
    metrics.append({
        'type': 'metrics',
        'dataset': dataset,
        'r2': r2,
        'mse': mse,
        'mad': mad
    })

# Cargar métrcias para observar los resultados
with gzip.open("files/models/model.pkl.gz", "rb") as file:
    loaded_model = pickle.load(file)
print(type(loaded_model))

for m in metrics:
    print(m)

# Convertir a tipos JSON-serializables
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Type {type(obj)} not serializable")
    
output_path = "files/output/metrics.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    for metric in metrics:
        json_line = json.dumps(metric, default=convert_to_serializable)
        f.write(json_line + "\n")

model_path = "files/models/model.pkl.gz"
try:
    with gzip.open(model_path, "rb") as file:
        model = pickle.load(file)
    print("Modelo cargado correctamente:", model)
except Exception as e:
    print("Error al cargar el modelo:", e)

print("Modelo entrenado y métricas guardadas correctamente.")


#print("Columnas de x_train:", x_train.columns)
#print("Columnas de x_test:", x_test.columns)

print(model.score(x_train, y_train))