# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:00:28 2024

@author: Juan
"""

# Regresion Lineal Multiple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importar dataset
dataset = pd.read_csv('Position_Salaries.csv')
# Utilizaremos solamente la columna level como var indep. ya que esta es prácticamente la columna position ya codificada
X = dataset.iloc[:, 1:2] 
''' Si utilizamoc X = dataset.iloc[:, 1] el resultado será tipo Serie (Vector) [1,2,3,4,5,6,7,8,9,10] mientras que
si utilizamos X = dataset.iloc[:, 1:2] la vairable contará con los mismos valores pero será tipo DataFrame
'''
y = dataset.iloc[:,2].values


# Ajustando una regresión lineal
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Ajustando una regresión plinomica
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)



# Visualizacion de los resultados del Modelo Lineal
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Modelo de Regression Lineal")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo en $")
plt.show()


# Visualizacion de los resultados del modelos Polinomico
X_grid = np.arange(min(X.values), max(X.values), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color= "blue")
plt.title("Modelo de Regression Polinomica")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo en $")
plt.show()


# Prediccion de nuestro modelo
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
