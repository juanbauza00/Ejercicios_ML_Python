# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 21:17:53 2024

@author: Juan
"""

# Plantilla de Pre Procesado

# Importacion de librerias
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Modificamos el directorio de nuestro proyecto
os.chdir("C:\\Users\\Juan\\Desktop\\Cursos\\ML de la A a la Z\\Archivos Python\\PreProcesadoDatos")

# Importar Dataset
dataset = pd.read_csv("Data.csv")

# Definimos nuestra variable dependiente(Y) e independiente (X)
# Utilizamos iloc (index localization para definir en nuestra variable el grupo de datos según el indice del dataset)
# Con iloc se especifica filas y columnas que vamos a tomar, al comenzar con : toma todas las filas
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3]

# Tratamiento de los Nan
# axis = 0 significa que va a tomar el promedio de los valores de la COLUMNA, si axis = 1 serían lo de la fila
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean");
#Se especifica que tome desde la columna indice = 1 hasta la indice = 2 (se coloca hasta el 3 ya que es hasta dicho numero sin incluir)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# Codificacion de datos categoricos

# Label encoder se utiliza para la codificacion de variables categoricas ordinales, en este caso solo se usa para ver su uso
labelEncoder = preprocessing.LabelEncoder()
X[:,0] = labelEncoder.fit_transform(X[:,0])
# ----------------------------------------------

# Para la codificacion de variables categoricas nominales (no poseen un orden entre ellas) se utiliza el metodo de variables Dummy
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])], remainder='passthrough' )
X = np.array(ct.fit_transform(X), dtype=int)

labelencoder_y = preprocessing.LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Dividir el dataset en conjunto de entrenamiento y de testing
# Random_state se utiliza como selmilla para que cada vez que ejecutemos el algoritmo nos seleccione los mismos datos de train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state= 0)


# Escalado de variables
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

