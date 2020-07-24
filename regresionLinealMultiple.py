"""  PRACTICA REGRESION LINEAL MULTIPLE  """
"""  Video: https://www.youtube.com/watch?v=kAfFxwiDvdQ&list=PLJjOveEiVE4Dk48EI7I-67PEleEC5nxc3&index=18 """

##### IMPORTAR LIBRERIAS A UTILIZAR ####

import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#### PREPARAR LOS DATOS ####

boston = datasets.load_boston()  # Importar una base de datos ya integrada con Scikit-Learn

#### ENTENDER LA BASE DE DATOS ####

print("Información en el Dataset: ")
print(boston.keys()) # Verifica la informacion contenida en el dataset.
print()

# Verificamos las caracteristicas del dataset
print("Caraterísticas del Dataset: ")
print(boston.DESCR) # Comando para Describir el conjunto de datos.
print()

# Verificamos la cantidad de datos que hay en los dataset
print("Cantidad de Datos: ")
print(boston.data.shape) # Comando para obtener cuantas (filas, columnas) tiene el dataset.
print()

# Verificamos la informacion en las columnas
print("Nombres de las columnas: ")
print(boston.feature_names) # Comando para obtener una lista con los nombres de cada columna.
print()

#### PREPARAR LA DATA PARA REGRESION LINEAL SIMPLE ####

# Seleccionamos las columnas 5, 6 y 7 del dataset
X_multiple = boston.data[:, 5:8]
print(X_multiple)

# Defino los datos correspondientes a las etiquetas (Variable dependiente)
y_multiple = boston.target

#### IMPLEMENTACION DE REGRESION LINEAL MULTIPLE ####

# Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos.
X_train, X_test, y_train, y_test = train_test_split(X_multiple, y_multiple, test_size = 0.2)

# Defino el algoritmo a utilizar
lr_multiple = linear_model.LinearRegression()

# Entreno el modelo
lr_multiple.fit(X_train, y_train)

# Finalmente, realizo una prediccion
Y_pred_multiple = lr_multiple.predict(X_test)

print("DATOS DEL MODELO DE REGRESION LINEAL MULTIPLE")
print()

print("Valor de las pendientes o coeficientes 'a': ")
print(lr_multiple.coef_)
print()

print("Valor de la interseccion o coeficiente 'b': ")
print(lr_multiple.intercept_)
print()

# Ahora calculamos la precision del algoritmo
print("Precision del modelo: ")
print(lr_multiple.score(X_train, y_train))