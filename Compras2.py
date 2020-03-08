# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:51:15 2019

@author: mvidal2
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap


# import some data to play with

db1=pd.read_csv('Data_UNAB.csv', sep=',' , parse_dates=['Fecha Aprobacion','Fecha Presentacion Reviewer','Fecha Validacion Presupuesto','Fecha creacion solicitud'], na_values=['0'])
db1.fillna("0")
#A = db1[(db1.Unidad de Negocio=='CHL01')]
#A3=db1.groupby(['Unidad de Negocio'])['ID_solicitud'].get_group(('CHL01')).unique()
A3=db1['ID_solicitud'].unique()
A4=A3.tolist()
dictionary = dict.fromkeys(A4, 0)


Cantidad_de_Lineas = []
count=0
#i=1
for i in range(len(db1['ID_solicitud'])):
    if db1['ID_solicitud'][i] in dictionary:
        #count=0
        count=dictionary.get(db1['ID_solicitud'][i])+1
        #Change Values dictionary
        dictionary[db1['ID_solicitud'][i]]=count
        #Accessing Items dictionary
        x = dictionary.get(db1['ID_solicitud'][i])
        Cantidad_de_Lineas.append(x)
    else:
        count=0
        #Change Values dictionary
        dictionary[db1['ID_solicitud'][i]]=count
        #Accessing Items dictionary
        x = dictionary.get(db1['ID_solicitud'][i])
        Cantidad_de_Lineas.append(x)
db1['Cantidad_de_Lineas'] = Cantidad_de_Lineas
######################################################################
def numeric_to_institucion(x):
    if x=='CHL04':
        return 1
    if x=='CHL32':
        return 2
    if x=='CHL02':
        return 3
    if x=='PER03':
        return 4
    if x=='PER05':
        return 5
    if x=='PER06':
        return 6
    if x=='CHL01' :
        return 7
    if x=='CHL05' or 'CHL08' or 'CHL06':
        return 8
    if x=='CHL18' or 'CHL25' or 'CHL28' or 'CHL31':
        return 9
    if x=='Sin institucion':
        return 10
    
db1['Unidad_Negocio_num'] = db1['Unidad de Negocio'].apply(numeric_to_institucion)
A=db1.reset_index(drop=True)
A=A.fillna("0")


A['Semana de creacion'] = A['Fecha creacion solicitud'].dt.week
A['Monto Total Solicitud Base'] = pd.to_numeric(A['Monto Total Solicitud Base'], errors='coerce')

C=A[(A['Unidad_Negocio_num']==7)&(A['Tipo presupuesto']=='OPEX')]


C1= C[['Monto Total Solicitud Base','Semana de creacion']]
kmeans = KMeans(init='k-means++',n_clusters=5, n_init=10)
kmeans.fit(C1)
y_kmeans = kmeans.predict(C1)

C1=C1.to_numpy() #conversion a Array


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = C1[:, 0].min() - .5, C1[:, 0].max() + .5
y_min, y_max = C1[:, 1].min() - .5, C1[:, 1].max() + .5
h = 200  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(C1[:, 0], C1[:, 1], cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()