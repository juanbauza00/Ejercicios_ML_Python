# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:55:34 2024

@author: Juan
"""

# Regresion Lineal Multiple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importar dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,4].values


# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
# Para la codificacion de variables categoricas nominales (no poseen un orden entre ellas) se utiliza el metodo de variables Dummy
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [3])], remainder='passthrough' )
X = np.array(ct.fit_transform(X), dtype=float)
# Evitamos la multicolinealidad generada por las variables dummy
X = X[:, 1:] #Eliminamos la primer columna
 

# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 0)

# Escalado de variables
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_Test = sc_X.fit_transform(X_Test)'''


# Ajustar el modelo de regresion lineal multiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)


# Prediccion de los resultados en el conjunto de testing
y_pred = regression.predict(X_test)





