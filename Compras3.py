# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 19:06:47 2019

@author: mvidal2
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.cluster import KMeans
import pandas as pd


db1=pd.read_csv('Data_UNAB.csv', sep=',' , parse_dates=['Fecha Aprobacion','Fecha Presentacion Reviewer','Fecha Validacion Presupuesto','Fecha creacion solicitud'], low_memory=False)
db1.fillna("0")
#A = db1[(db1.Unidad de Negocio=='CHL01')]
#A3=db1.groupby(['Unidad de Negocio'])['ID_solicitud'].get_group(('CHL01')).unique()
A3=db1['ID_solicitud'].unique()
A2=db1['Código subcategoría'].unique()

A4=A3.tolist()
A1=A2.tolist()

dictionary = dict.fromkeys(A4, 0)
dictionary2 = dict.fromkeys(A1, 0)


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

##############################################################################
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
    
##############################################################################
        
db1['Unidad_Negocio_num'] = db1['Unidad de Negocio'].apply(numeric_to_institucion)
A=db1.reset_index(drop=True)
A=A.fillna("0")

A['Fecha creacion solicitud'] = pd.to_datetime(A['Fecha creacion solicitud'], errors='coerce')

A['Semana de creacion'] = A['Fecha creacion solicitud'].dt.week
A['Monto Línea Solicitud (Moneda Base)'] = pd.to_numeric(A['Monto Línea Solicitud (Moneda Base)'], errors='coerce')


counter_list = list(enumerate(A2,1))
type(counter_list)
tabla=pd.DataFrame(counter_list)
tabla.columns = ['conteo', 'subcategoria']
tabla2=tabla[['subcategoria','conteo']]
tabla2
TABLA2=tabla2['subcategoria'].tolist()
TABLA22=tabla2['conteo'].tolist()
subcategory = zip(TABLA2, TABLA22)
dictionary_subcategory = dict(subcategory)
A['Subcategoria_Numerica']=A['Código subcategoría'].map(dictionary_subcategory)

##############################################################################

"""
C=A[(A['Unidad_Negocio_num']==7)&(A['Tipo presupuesto']=='OPEX')]
C1= C[['Monto Total Solicitud Base','Semana de creacion']]
C2 = C[['Monto Total Solicitud Base','Semana de creacion']]
kmeans = KMeans(init='k-means++',n_clusters=5, n_init=10)
kmeans.fit(C1)
y_kmeans = kmeans.predict(C1)
C1=C1.to_numpy() #conversion a Array
"""
A=A[(A['Unidad_Negocio_num']==7)&(A['Tipo presupuesto']=='OPEX')&(A['Monto Línea Solicitud (Moneda Base)']<40000000)]

A.to_excel("revisar.xlsx")

X = A[['Monto Línea Solicitud (Moneda Base)','Semana de creacion']]
y = A[['Subcategoria_Numerica']]

X=X.loc[:8000]
y=y.loc[:8000]

X=X.to_numpy()
y=y.to_numpy()
y=y.ravel()

##############################################################################
from sklearn import metrics
                            
n_neighbors=3 #el mejor es 30 => 24%
h=6

for weights in ['distance']: #['uniform', 'distance']
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)
    y_pred = clf.predict(X)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z,cmap='Pastel1')
    #plt.pcolormesh(xx, yy, Z,cmap=plt.cm.Greens)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c='r',edgecolors='k', cmap=plt.cm.coolwarm)
    #plt.xlim(xx.min(), xx.max())
    #plt.ylim(yy.min(), yy.max())
   
    Accuracy = round(metrics.accuracy_score(y, y_pred)*100,2)
    
    plt.title("3-Class classification (k = %i, weights = '%s', Predictor Neigbors = '%s')"
              % (n_neighbors, weights, Accuracy))

plt.show()


##############################################################################

from sklearn.neighbors import KNeighborsClassifier


k_range = list(range(1, 80))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    scores.append(metrics.accuracy_score(y, y_pred))


plt.scatter(k_range, scores, c='green',marker="s", s=5)
plt.title('Evaluacion del mejor K',fontsize=10)
plt.xlabel('Value of K for KNN',fontsize=8)
plt.ylabel('Testing Accuracy',fontsize=8)

##############################################################################

"""
Realizando division del dataset
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=2)

"""
KNN = 5 con dataset dividido
"""
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

##############################################################################

#n_neighbors=3 #el mejor es 30 => 24%
#h=9
weights='distance'

for algorithm in ['auto']: #['auto','ball_tree','kd_tree']
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights, algorithm=algorithm)
    clf.fit(X, y)
    y_pred2= clf.predict(X)
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z,cmap='Pastel1')
    #plt.pcolormesh(xx, yy, Z,cmap=plt.cm.Greens)

    
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c='r',edgecolors='k', cmap=plt.cm.coolwarm)
    #plt.xlim(xx.min(), xx.max())
    #plt.ylim(yy.min(), yy.max())

    Accuracy2 = round(metrics.accuracy_score(y, y_pred2)*100,2)
    plt.tight_layout()
    plt.title("3-Class classification (k = %i, algorithm = '%s', Predictor Neigbors = '%s')"
              % (n_neighbors, algorithm, Accuracy))

plt.show()

##############################################################################

"""
Centroides
"""
for k in [3]:
    
    kmeans = KMeans(init='k-means++',n_clusters=k).fit(X)
    centroids = kmeans.cluster_centers_

    plt.scatter(A['Monto Línea Solicitud (Moneda Base)'], A['Semana de creacion'], s=5, cmap='plasma')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black',marker="x", s=80)
    plt.title('Centroides = 3',fontsize=10)
    
plt.show()


##############################################################################

for k in [3]:
    
    kmeans = KMeans(init='k-means++',n_clusters=k, n_init=10)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    centers = kmeans.cluster_centers_
    
    plt.scatter(A['Monto Línea Solicitud (Moneda Base)'], A['Semana de creacion'], s=5, cmap='plasma')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black',marker="x", s=80)
    plt.title('Centroides = 3',fontsize=10)

plt.show()


##############################################################################

