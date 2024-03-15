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


# Construir el modelo optimo de RLM utilizando la Eliminacion hacia atras
# Preparación para el modelo utilizando la Eliminación hacia atrás
import statsmodels.api as sm

# Notas: ------------------------------------------------------
'''
La modelación de un modelo otpimo de Regresion Lineal Multiple (RLM) utilizando el método de 
Eliminación hacia atrás se basa en:
    PASO 1: Seleccionar el nivel de significación para permanecer en el modelo (p.e SL=0.05)
    PASO 2: Se calcula el modelo con todas las variable predictoras
    PASO 3: Se analiza la variable predictora con el p-value mas grande. Si P > SL, entoces
    vamos al paso 4, sino vamos a FIN
    PASO 4: Se elimina la variable predictora.
    PASO 5: Se ajusta nuevamente el modelo (se genera de nuevo desde 0 sin la variable eliminada)
    y luego de ajustar el modelo volvemos al PASO 3.


Vamos a pensar en la formula de una RLM, la variable target 'y' es resultado de una serie de sumas
de UN TERMINO INDEPENDIENTE (constante) y varias variables independientes acompañadas de un coeficiente
que los multiplica. y = b0 + b1*x1 + b2*x2 + ... bn*xn

Lo primero que se debe hacer es identificar los p-values del los coeficientes ya que si su p-value
es cercano a 1, es muy posible que tome un valor cercano a 0 (la varible resulta irrelevante)
y el p-value del término independiente debido a que este tambien influye estadisticamente sobre el modelo.

La forma propuesta para calcular este p-value es agregar una columna adicional de unos en la matriz (data frame)
para que el término independiente sea explícitamente modelado. Luego, el p-value del coeficiente
correspondiente a esta columna adicional se puede analizar para determinar su relevancia estadística.

'''

# Añadimos columna de 1
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)  

# PASO 1
SL = 0.05

# PASO 2
# Creamos una variable que va a almacenar las variables estadisticamente significativas para el modelo
# Como estamos utilizando el metodo de eliminación hacia atrás se le asignan todas las variables a X_opt y 
X_opt = X[:, [0, 1, 2, 3, 4, 5]] 

# debido a que la funcion de sm espera una lista de listas, se transforma X_opt (que es un array n-dimensional) a .tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit() # (OLS = Ordinary Least Squares)
regression_OLS.summary() # Revisamos los p-values

# PASO 3
'''
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.013e+04   6884.820      7.281      0.000    3.62e+04     6.4e+04
x1           198.7888   3371.007      0.059      0.953   -6595.030    6992.607
x2           -41.8870   3256.039     -0.013      0.990   -6604.003    6520.229
x3             0.8060      0.046     17.369      0.000       0.712       0.900
x4            -0.0270      0.052     -0.517      0.608      -0.132       0.078
x5             0.0270      0.017      1.574      0.123      -0.008       0.062

Podemos ver que de momento las variables MAS RELEVANTES son const (termino independiente) y la columna nro 3 (R&D Spend)
mientras que las variable MENOS RELEVANTES, con un p-value cercano a 1 son la columna nro 2 y nro 1.
Siguiendo al PASO 4 eliminaremos la variable con el mayor p-vale (col nro 2, p-val = 0.99)
'''


# PASO 4 (eliminamos col nro 2)
X_opt = X[:, [0, 1, 3, 4, 5]] 
# PASO 5
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary() # Revisamos los p-values

# Volvemos al PASO 3 y realizamos todos los pasos hasta finalizar
X_opt = X[:, [0, 3, 4, 5]] 
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

# 2do loop
X_opt = X[:, [0, 3, 5]] 
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

# 3er loop
X_opt = X[:, [0, 3]] 
regression_OLS = sm.OLS(endog = y, exog = X_opt.tolist()).fit()
regression_OLS.summary()

'''
NOTA:
    El criterio de Akaike (AIC) y el criterio Bayesiano (BIC) son medidas utilizadas para
    la selección de modelos en estadística, específicamente en el contexto de regresión.
    Estas medidas penalizan la complejidad del modelo, favoreciendo aquellos modelos
    que tienen un buen ajuste a los datos pero que no son excesivamente complejos.
    
    El AIC (en terminos simples) cuantifica la cantidad de información perdida por un modelo dado,
    cuanto menor sea el valor de AIC, mejor es el modelo.
    El BIC (en terminos simples) también penaliza la complejidad del modelo, pero de manera más fuerte que el AIC.
    
Analisis:
    Siendo estrictos con nuestro SL y sumando el hecho de que tanto el AIC como el BIC decrecen al
    eliminar la columna nro 5, podemos decir que el modelo que mejor se ajusta a las predicciones es
    una regresión lineal simple quedando únicamente con el término independiente y la variable
    de la columna 3 (R&D Spend)
'''

# Automatizando el procedimiento (solamente teniendo en cuenta los p-values)
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)



# Automatizando el procedimiento (teniendo en cuenta los p-values y el Adj. R-squared)
def backwardElimination(x, SL):    
    numVars = len(x[0])    
    temp = np.zeros((50,6)).astype(int)    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):                        
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL) #En este caso el modelo mantiene la ordenada y las columnas 3 y 5
