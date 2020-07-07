"""  PRACTICA REGRESION LINEAL SIMPLE  """
"""  Video: https://www.youtube.com/watch?v=YLGhVBB5rGU&list=PLJjOveEiVE4Dk48EI7I-67PEleEC5nxc3&index=16 """

##### IMPORTAR LIBRERIAS A UTILIZAR ####

import numpy as np
from sklearn import datasets, linear_model
#from sklearn.model_selection import train_test_split
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

# Seleccionamos una sola columna del dataset como nuestra variable independiente
X = boston.data[:, np.newaxis, 9] # Columna 9, Impuestos.

# Defino los datos correspondientes a las etiquetas como variable dependiente
y = boston.target # Selecciono el precio de las casas

# Graficamos los datos usando MatPlotLib
plt.scatter(X, y)
plt.xlabel("Impuestos")
plt.ylabel("Valor medio")
plt.title("Grafico de los Datos Esparcidos")
plt.show()


#### IMPLEMENTACION DE REGRESION LINEAL SIMPLE #### 

from sklearn.model_selection import train_test_split  # Importar libreria

# Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Defino el Algoritmo a utilizar
lr = linear_model.LinearRegression()

# Entreno el modelo
lr.fit(X_train, y_train)

# Realizo una prediccion
Y_pred = lr.predict(X_test)

# Muestro en pantalla los resultados obtenidos. Grafico los datos junto el modelo
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color = "green", linewidth = 3)
plt.title("Regresion Lineal Simple")
plt.xlabel("Impuesto")
plt.ylabel("Valor medio")
plt.title("Predicción generada por el algoritmo")
plt.show()

## Se observa que este algoritmo no seria el mejor para este tipo de datos que estan tan severamente esparcidos

# Mostremos la pendiente, veamos como quedo nuestra ecuacion
print()
print("DATOS DEL MODELO DE REGRESIÓN SIMPLE")
print()
print("Valor de la pendiente o coeficiente 'a': ")
print(lr.coef_) # Comando para obtener la pendiente
print("Valor de la intersección o coeficiente 'b': ")
print(lr.intercept_) # Comando para obtener la interseccion
print()
print("La ecuación del modelo es igual a: y = " + str(lr.coef_) + "x + " + "(" + str(lr.intercept_) + ")")
print()

# Calculemos la precision del algoritmo
print("Precisión del modelo: ")
print(lr.score(X_train, y_train))