# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 22:39:15 2024

@author: Juan
"""
# SVR
# Librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importación del DataSet
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = y.reshape(-1, 1)
y = sc_y.fit_transform(y)

# Ajustar la regresion con el dataset
from sklearn.svm import SVR
regression = SVR(kernel = "rbf")
regression.fit(X,y)

# Prediccion de nuestros modelos con SVR
y_pred = regression.predict([[6.5]])

# Visualizacion de los resultados del SVR
plt.scatter(X, y, color = "red")
plt.plot(X, regression.predict(X), color= "blue")
plt.title("Modelo de Regression (SVR)")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo en $")
plt.show()







