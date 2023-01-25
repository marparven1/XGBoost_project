# -*- coding: utf-8 -*-
"""
Clasificador utilizando XGBosot

Primer utilizamos el modelo de skleran de XGBoost para regresión
Dataset: Computer Hardware Data Set: Medidas del rendimiento relativo y las características de 209 CPUs
@author: Marta Venegas
"""
#
# CONJUNTO DE DATOS: cpus.txt

#%% Librerías 
import pandas as pd
# from xgboost import plot_tree
# from graphviz import Source
import matplotlib.pyplot as plt
import numpy as np

from xgboost import XGBRegressor
print("Documentación de XGBRegressor:\n")
print(XGBRegressor.__doc__)
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV



#%% Lectura de los datos 
print("-"*50)
print("Lectura de datos: \n")
data = pd.read_csv("cpus.txt",delimiter = "\s+",index_col=False)
print(data.head())


#%% Descripción de los datos
print("-"*50)
print("Desripción de las variables:")
#print(data.columns)


dic = {
    'case' : 'número de caso',
'name' : 'fabricante y modelo.',
'syct' : 'tiempo de ciclo en nanosegundos.',
'mmin' : 'memoria principal mínimo en kilobytes.',
'mmax' :' memoria principal míxima en kilobytes.',
'cach' : 'tamaño de caché en kilobytes.',
'chmin' : 'número mínimo de canales.',
'chmax' : 'número míximo de canales.',
'perf' : 'rendimiento publicado con relación a un 370/158-3 IBM.',
'estperf' : 'rendimiento estimado (en Ein-Dor y Feldmesser).'
}

for i in dic:
    print("\n-",i,':',dic[i])
print("-"*50)
print("Descripción de los datos:")
print(data.describe())
print("-"*50)
print("Información de los datos:")
print(data.info())
print("No hay datos faltantes.")
print("En total, tenemos",data.shape[0],"registros (muestras) y ",data.shape[1]," variables.")
print("-"*50)
print("Nuestro objetivo es predecir la variable target: Y=PERF, que mide el rendimiento publicado con relación a un 370/158-3 IBM.Para umplir nuestro objetivo, haremos uso del resto de features, salvo la variable NAME, que nos da información sobre el fabricante y modelo de la CPU.")


#%% División en entrenamiento y test
print("-"*50)
print("División de los datos de entrenamiento y testeo con una proporción: 80,20")
X =data.drop(["PERF","NAME"],axis=1)
y = data['PERF']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
print("Datos de entrenamiento: ",X_train.shape)
print("Datos de testeo: ",X_test.shape)

#%%
print("-"*50)
print("Definimos el modelo por defecto:\n")
modeloXGBoost = XGBRegressor(random_state=42)
print(modeloXGBoost)


#%% Posibles hiperparámetros de este modelo

print("-"*50)
print("#"*15,"HIPERPARÁMETROS DEL MODELO XGBOOST","#"*15)
HIPER_DIC = {"objetive":"Función de pérdida para la optimización del modelo",
    "n_estimators":" Número de estimadores (árboles en el modelo)",
            "max_depth":"Máxima profundidad de los árboles",
            "eta": "Tasa de aprendizaje: learning rate ",
            "gamma": "Parámetro de regularización que ayuda a cotrolar el sobreajuste. especifica una penalización para las hojas del árbol que tienen una pérdida reducida. .Conforme mayor es, mas conservativo el algoritmo",
            "lambda":"Término de regularización L2 : a mayor, mas conservador",
            "alpha":"Término de regularización L1 : a mayor, mas conserva. Usado para dataset de grandes dimensiones",
            "early_stopping_rounds":"Número de iteraciones antes de detener el entrenamiento temprano si no se observa mejora en el rendimiento",
            "subsample": "Proporción de muestras utilizadas para caada árbol",
            "colsample_bytree":"Proporción de características utilizadas para cada árbol",
            "eval_metric": "Métrica para evaluar el rendimiento del modelo"
            }
# En XGBoost, el hiperparámetro gamma es un parámetro de regularización que ayuda a controlar el sobreajuste. Este hiperparámetro especifica una penalización para las hojas del árbol que tienen una pérdida reducida.

# La idea detrás de esto es que, si una hoja del árbol tiene una pérdida muy baja, significa que el modelo está muy seguro de las predicciones en esa hoja, lo que podría indicar un sobreajuste. Por lo tanto, utilizando un valor alto para el hiperparámetro gamma penalizará esas hojas con pérdida baja, lo que ayudará a reducir el sobreajuste en el modelo.

for i in HIPER_DIC:
    print("\n- {}:{} ".format(i,HIPER_DIC[i]))

#%%
print("-"*50)
print("Entrenamiento del modelo con los métodos: \n -.fit() \n - .predict() \n- .r2_score()")
modeloXGBoost.fit(X_train, y_train)
y_pred_train =modeloXGBoost.predict(X_train)
R2_entrenamiento = metrics.r2_score(y_train,y_pred_train )
print('R2 para el entrenamiento:',R2_entrenamiento)




#%%
print("-"*50)
print("Predicciones: \n - Método .predict()")
y_pred = modeloXGBoost.predict(X_test)


MSE = np.sqrt( MSE(y_test, y_pred) )
print("- MSE: %.5f" % MSE )
R2_test = metrics.r2_score(y_test, y_pred)
R2= modeloXGBoost.score(X_test, y_test) # es lo mismo
print('R2 en el testeo:',R2_test)


#%% Feature importance
print("-"*50)
# Dibujamos la importancia de las variables predictoras en el modelo
feature_importances = pd.Series(modeloXGBoost.feature_importances_, index=X.columns)
feature_importances.plot(kind='barh')
print("\nImportancia de las variables en el modelo base (en tanto por ciento)\n")
for name, score in zip(X.columns, modeloXGBoost.feature_importances_):
    print('-',name,':', score*100)



#%%
print("-"*50)
print("Definición de un modelo óptimo con una malla de hiperparametrización")
print("\n Malla de hiperparámetros:")
malla = {#"objective":["reg:squarederror","reg:linear"], 
            "n_estimators": [50,100,150,200], # Número de estimadores (árboles en el modelo)
            "max_depth":[3,5,7], # Máxima profundidad de los árboles
            "eta":[0.01,0.1], # Tasa de aprendizaje: learning rate 
            #"gamma":[], # Conforme mayor es, mas conservativo el algoritmo
            # "lambda":[], # Término de regularización L2 : a mayor, mas conservador
            # "alpha":[], # Término de regularización L1 : a mayor, mas conserva. Usado para dataset de grandes dimensiones
            # "early_stopping_rounds" Número de iteraciones antes de detener el entrenamiento temprano si no se observa mejora en el rendimiento
            "subsample":[0.1,0.3,0.5,0.7], # Proporción de muestras utilizadas para caada árbol
            "colsample_bytree":[0.4,0.6,0.8] ,# Proporción de características utilizadas para cada árbol
            "eval_metric":["rmse","mae","logloss","auc"] # Métrica para evaluar el rendimiento del modelo
            
            }

for i in malla:
    print("\n-",i,':',malla[i])
    
print("-"*50)
print("\nDefinimos el modelo con la malla y con CV:")

gsc = GridSearchCV(
            estimator=XGBRegressor(), # Algoritmo
            param_grid=malla, # Malla con hiperparámetros
            cv=3, # 3 folds de CV porque hay pocos datos
            verbose=1,
            n_jobs=-1 # Uso todos los procesadores
            )
print(gsc)

import time
t1 = time.time()                          
gsc.fit(X_train, y_train)
t2 = time.time()
t_grid = t2 - t1
print("\nTiempo búsqueda grid:", t_grid, " segundos.")
print("-"*50)
print("\nMejores parámetros:")
print('\n - objective: reg:squarederror')
for i in gsc.best_params_:
    print("\n-",i,':',gsc.best_params_[i])

eta = gsc.best_params_.get('eta')
max_depth = gsc.best_params_.get('max_depth')
n_est = gsc.best_params_.get('n_estimators')
colsample_bytree = gsc.best_params_.get('colsample_bytree')
subsample = gsc.best_params_.get('subsample')
eval_metric = gsc.best_params_.get('eval_metric')
#objetive= gsc.best_params_.get('objective')


mejor_modelo =  XGBRegressor(random_state=42,
                            # objective =objetive,
                            objective="reg:squarederror",
                             eta=eta,
                             max_depth=max_depth,
                             n_estimators=n_est,
                             colsample_bytree=colsample_bytree,
                             eval_metric=eval_metric,
                             subsample=subsample
                             )
# print(mejor_modelo)

#%%
print("-"*50)
print("Entrenamos el modelo configurado con los mejores hiper parámetros con el método .fit() ")
# Estimamos los parámetros del modelo a partir de los datos de entrenamiento
mejor_modelo.fit(X_train, y_train)
print("A continuación, calculamos las predicciones con el método .predict()")
y_pred_train =mejor_modelo.predict(X_train)
print("Por último, calculamos la métrica R2, coeficiente de correlación. Este coeficiente mide el porcentaje de la variabilidad de la variable respuesta que el modelo capta.")
R2_entrenamiento = metrics.r2_score(y_train,y_pred_train )
print('R2 para el entrenamiento:',R2_entrenamiento)




#%%
print("-"*50)
print("Predicciones para los datos de testeo con el método .predict()")
# Predicciones
y_pred = mejor_modelo.predict(X_test)
print("Evaluamos la calidad de las predicciones con el método .score()")
# Evaluamos la calidad de las predicciones
R2_test = metrics.r2_score(y_test, y_pred)
R2= mejor_modelo.score(X_test, y_test) # es lo mismo
print('R2 en el testeo:',R2_test)
print("El modelo capta un {} \% de la variabilidad total de la respuesta, es decir, del rendimiento publicado con relación a un 370/158-3 IBM ".format(R2_test*100))

#%% Feature importance
print("-"*50)

# Dibujamos la importancia de las variables predictoras en el modelo
feature_importances = pd.Series(mejor_modelo.feature_importances_, index=X.columns)
feature_importances.plot(kind='barh')
plt.show()
print("Importancia de las variables en el modelo:")
for name, score in zip(X.columns, mejor_modelo.feature_importances_):
    print('-',name,':', score)

    
# print("Vamos a visualizar el árbol del mejor modelo:")

# plot_tree(mejor_modelo,rankdir='LR')
# plt.savefig("tree_structure.pdf")
# plt.show()
# from sklearn import tree
# tree.export_graphviz(mejor_modelo,
#                      out_file="BEST_tree.dot" ,
#                     feature_names = ['x1','x2','x3','x4','x5','x6','x7','x8'], 
#                      class_names= ['y'],
#                      filled = True) # Con graphviz

# Source.from_file("BEST_tree.dot")