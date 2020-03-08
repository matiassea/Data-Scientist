# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:30:59 2019

@author: mvidal2
http://scott.fortmann-roe.com/docs/BiasVariance.html

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


start1 = time.time()
db1=pd.read_csv('Data_UNAB.csv', sep=',' , parse_dates=['Fecha Aprobacion','Fecha Presentacion Reviewer','Fecha Validacion Presupuesto','Fecha creacion solicitud'], na_values=['0'],low_memory=False)
db1.fillna("0")
db1=db1.loc[:,['Tipo presupuesto','Monto Total Solicitud Base','Fecha creacion solicitud','Cantidad_de_Lineas', 'Unidad_Negocio_num','Unidad de Negocio','ID_solicitud','CECO','Código subcategoría','ID_solicitante']]
#db1=db1.column_index(df, ['peach', 'banana', 'apple'])(drop=True)
db1=db1.set_index(['Fecha creacion solicitud'])

"""
db.get_dtype_counts()
db['ID_solicitud'].unique()
db['ID_solicitud'].value_counts()
print(db.columns)
print(db.head(3))
print(db.dtypes)

#Obtener descripciones por tipo de datos.
print(db.describe(include=['object']))
print(db.describe(include=['int64']))
print(db.describe(include=['float64']))
print(db['Fecha Presentacion Reviewer'].dtype)
print(db['Fecha Validacion Presupuesto'].dtype)

#Creacion de columna
#db['F_Contable'] = db.F_Contable.astype(datetime)
db['week'] = db['Fecha Aprobacion'].dt.week
"""

  
"""
http://benalexkeen.com/resampling-time-series-data-with-pandas/

db['SLA1'] = db['Fecha Presentacion Reviewer']-db['Fecha creacion solicitud']
db['SLA2'] = db['Fecha Validacion Presupuesto']-db['Fecha creacion solicitud'] 

db['SLA1'] = db['Fecha Presentacion Reviewer'].dt.day-db['Fecha creacion solicitud'].dt.day
db['SLA2'] = db['Fecha Validacion Presupuesto'].dt.day-db['Fecha creacion solicitud'] .dt.day

print(db['SLA1'].dtype)
print(db['SLA2'].dtype)

"""


# converting to dict 
#data_dict = db['ID solicitud'].to_dict() 
#sns.jointplot(x='SLA1', y='SLA2',data=db,kind='reg')


ID=db1['ID_solicitud'].nunique()

print(ID, "Total de solicitudes")
#print(db.nunique())

A=db1.groupby(['Unidad de Negocio'])['ID_solicitud'].get_group(('CHL01')).nunique()
print(A, "Solicitudes UNAB")

B=db1.groupby(['Unidad de Negocio'])['CECO'].get_group(('CHL01')).nunique()
print(B," CeCos")

C=db1.groupby(['Unidad de Negocio'])['Código subcategoría'].get_group(('CHL01')).nunique()
print(C," subcategorias")

D=db1.groupby(['Unidad de Negocio'])['ID_solicitante'].get_group(('CHL01')).nunique()
print(D, " Requestors")

end1 = time.time()

print(end1 - start1, " Tiempo de comando")


start2 = time.time()


Quantity_Request_line=db1['ID_solicitud'].resample('B').count()
Quantity_Request_line=pd.Series(Quantity_Request_line).values
Quantity_Request=db1['ID_solicitud'].resample('B').nunique()
Quantity_Request=pd.Series(Quantity_Request).values
sns.jointplot(x=Quantity_Request_line, y=Quantity_Request,kind="scatter",space=0,).set_axis_labels("Quantity_Request_line", "Quantity_Request")

   
#db1=pd.read_csv('Data_UNAB.csv', sep=',' , parse_dates=['Fecha Aprobacion','Fecha Presentacion Reviewer','Fecha Validacion Presupuesto','Fecha creacion solicitud'], na_values=['0'])
#db1.fillna("0")
#A = db1[(db1.Unidad de Negocio=='CHL01')]
#A3=db1.groupby(['Unidad de Negocio'])['ID_solicitud'].get_group(('CHL01')).unique()
A3=db1['ID_solicitud'].unique()
"""
#https://docs.python.org/3.3/tutorial/datastructures.html
#https://chrisalbon.com/python/data_wrangling/create_list_from_dictionary_keys_and_values/
#https://chrisalbon.com/python/data_wrangling/pandas_create_column_with_loop/
#https://docs.python.org/3.3/tutorial/datastructures.html#using-lists-as-queues
https://thispointer.com/python-how-to-convert-a-list-to-dictionary/
https://www.w3schools.com/python/python_dictionaries.asp



Array to list
ndarray.tolist()
Return the array as a (possibly nested) list.
https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tolist.html


List to Dict
https://thispointer.com/python-how-to-convert-a-list-to-dictionary/
Converting a list to dictionary with list elements as keys in dictionary
using dict.fromkeys()

The fromkeys() method creates a new dictionary from the given sequence of elements with a value provided by the user. 


sequence - sequence of elements which is to be used as keys for the new dictionary
value (Optional) - value which is set to each each element of the dictionary



dictionary = dict.fromkeys(A4, 1)
print(dictionary)


Dict
Loop Through a Dictionary => for x in thisdict: print(x)
When looping through a dictionary, the return value are the keys of the dictionary

Loop Through a Dictionary => for x in thisdict.values(): print(x)
When looping through a dictionary, the return value are the value of the dictionary

Loop through both keys and values, by using the items() function:
for x, y in thisdict.items():
    print(x, y) 
    
Check if Key Exists
To determine if a specified key is present in a dictionary use the in keyword:
if "model" in thisdict:
print("Yes, 'model' is one of the keys in the thisdict dictionary") 



Change Values => thisdict["year"] = 2018
You can change the value of a specific item by referring to its key name:

Accessing Items => x = thisdict["model"]    
You can access the items of a dictionary by referring to its key name, inside square brackets:

"""
######################################################################
A4=A3.tolist()
dictionary = dict.fromkeys(A4, 0)
######################################################################
end2 = time.time()
print(end2 - start2, " Tiempo de dictionary")

######################################################################

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

#########################################################################
A=db1.reset_index()
#A=db1.reset_index(drop=True)
A=A.fillna("0")
A=A[(A['Unidad_Negocio_num']==7)&(A['Tipo presupuesto']=='OPEX')]
#A=A[(A['Unidad_Negocio_num']==7)&(A['Tipo presupuesto']=='OPEX')&(A['Monto Línea Solicitud (Moneda Base)']<40000000)]
#########################################################################

A['Fecha creacion solicitud'] = pd.to_datetime(A['Fecha creacion solicitud'], errors='coerce')
A['Semana de creacion'] = A['Fecha creacion solicitud'].dt.week

#########################################################################

#https://datatofish.com/convert-string-to-float-dataframe/
A['Monto Total Solicitud Base'] = pd.to_numeric(A['Monto Total Solicitud Base'], errors='coerce')
#A['Monto Total Solicitud Base'] = A['Monto Total Solicitud Base'].astype(float)
#########################################################################

A.to_excel("maseva2.xlsx")


#########################################################################

A1=A[['Monto Total Solicitud Base','Cantidad_de_Lineas']]
A2=A['Semana de creacion']

print(A1.shape," Tamaño matriz A1")
print(A2.shape," Tamaño matriz A2")

print(type(A1)," Tipo matriz A1")
print(type(A2)," Tipo matriz A2")

A1=A1.to_numpy()
A2=A2.to_numpy()

"""
Sum list of dictionaries with same key
https://www.geeksforgeeks.org/python-sum-list-of-dictionaries-with-same-key/
"""



"""
https://stackoverflow.com/questions/31076698/create-a-pandas-dataframe-of-counts
https://stackoverflow.com/questions/22391433/count-the-frequency-that-a-value-occurs-in-a-dataframe-column
https://www.eumus.edu.uy/eme/ensenanza/electivas/python/CursoPython_clase11.html
https://github.com/justmarkham/scikit-learn-videos

"""
"""
Logistic Regression
"""

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(multi_class='auto',solver='liblinear')
logreg.fit(A1,A2)
y_pred = logreg.predict(A1)
print(len(y_pred)," Largo del predictor A1")

from sklearn import metrics
#es igual a print(logreg.score(A1,A2)*100, " score")
print(metrics.accuracy_score(A2, y_pred)*100," % predictor Logistic Regression")

x_min,x_max = A1[:, 0].min() - .5, A1[:, 0].max() + .5
y_min, y_max = A1[:, 1].min() - .5, A1[:, 1].max() + .5
h = 10000  # step size in the mesh

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


fig=plt.figure(figsize=(10,10)) # 10 is width, 10 is height
ax1=fig.add_subplot(4,2,1)



plt.scatter(A1[:, 0], A1[:, 1], c='G', cmap=plt.cm.Paired)
ax1.pcolormesh(xx, yy, Z,cmap=plt.cm.Paired)
ax1.set_xlabel('Monto Total Solicitud Base',fontsize=10)
ax1.set_ylabel('Cantidad_de_Lineas',fontsize=8)
ax1.set_title('Scatter',fontsize=10)
#ax1.set_xlim(xx.min(), xx.max())
#ax1.set_ylim(yy.min(), yy.max())



"""
PCA
"""
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(A1)
X = pca.transform(A1)

#X = pca.fit_transform(A1)

ax2=fig.add_subplot(4,2,2)
plt.scatter(X[:, 0], X[:, 1], s=5,marker="o")
ax2.set_xlabel('Monto Total Solicitud Base',fontsize=8)
ax2.set_ylabel('Cantidad_de_Lineas',fontsize=8)
ax2.set_title('PCA components=2',fontsize=10)
ax2.set_xlim((0, 20000000))
ax2.set_ylim((0, 70))


"""
Lectura de graficos
https://www.machinelearningplus.com/plots/matplotlib-tutorial-complete-guide-python-plot-examples/
https://realpython.com/python-matplotlib-guide/
https://github.com/matiassea/Machine-Learning-with-Python/blob/master/Machine%20Learning%20for%20Diabetes.ipynb

"""
ax3=fig.add_subplot(4,2,3)
#plt.scatter(B1['Monto Total Solicitud Base'], B1['Cantidad_de_Lineas'], c= kmeans.labels_.astype(float), s=5,marker="o", alpha=0.5)
plt.scatter(A1[:, 0], A1[:, 1], s=5,marker="o", alpha=1)
ax3.set_xlim((0, 20000000))
ax3.set_ylim((0, 70))
ax3.set_title('Grafica de la data',fontsize=10)
ax3.set_xlabel('Monto Total Solicitud Base',fontsize=8)
ax3.set_ylabel('Cantidad_de_Lineas',fontsize=8)



"""
Neigbors = 5
"""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(A1,A2)
y_pred = knn.predict(A1)
print(metrics.accuracy_score(A2, y_pred)*100," % predictor Neigbors = 5")

"""
Monto Total solicitud Base vs Cantidad de Lineas
"""

B1= A[['Monto Total Solicitud Base','Cantidad_de_Lineas']]
B1.get_values
B1.shape



"""
Centroides
"""
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5).fit(B1)

centroids = kmeans.cluster_centers_

ax4=fig.add_subplot(4,2,4)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red',marker="x", s=50)
ax4.set_title('Centroides = 5',fontsize=10)


"""
Realizando division del dataset
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(A1,A2,test_size=0.4,random_state=2)
print(X_train.shape, "=x train", y_train.shape, "=y train", X_test.shape, "=x test", y_test.shape, "=y test")

"""
KNN = 5 con dataset dividido
"""
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred)*100," % predictor Neigbors = 5, con dataset dividido")

"""
LogisticRegression con dataset dividido
"""
logreg = LogisticRegression(multi_class='auto',solver='liblinear')
logreg.fit(X_train,y_train)
y_pred2 = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred2)*100," % predictor Logistic Regression, con dataset dividido")


"""
Evaluando el mejor valor de K
"""
k_range = list(range(1, 80))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

ax5=fig.add_subplot(4,2,5)
#ax4.plt.plot(k_range, scores)
plt.scatter(k_range, scores, c='green',marker="s", s=5)
ax5.set_title('Evaluacion del mejor K',fontsize=10)
ax5.set_xlabel('Value of K for KNN',fontsize=8)
ax5.set_ylabel('Testing Accuracy',fontsize=8)


from sklearn.cluster import KMeans
kmeans = KMeans(init='k-means++',n_clusters=5, n_init=10)
kmeans.fit(B1)
y_kmeans = kmeans.predict(B1)
centers = kmeans.cluster_centers_



"""
fig.add_subplot(ROW,COLUMN,POSITION)
ROW=number of rows
COLUMN=number of columns
POSITION= position of the graph you are plotting 
https://stackoverflow.com/questions/3584805/in-matplotlib-what-does-the-argument-mean-in-fig-add-subplot111/9850790
https://stackoverflow.com/questions/51111762/customizing-subplots-in-matplotlib

"""
ax6=fig.add_subplot(4,2,6) 
#plt.scatter(B1['Monto Total Solicitud Base'], B1['Cantidad_de_Lineas'], c=y_kmeans, s=10, cmap='plasma')
plt.scatter(B1['Monto Total Solicitud Base'], B1['Cantidad_de_Lineas'], s=5, cmap='plasma')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=80, alpha=1)
ax6.set_title('Centroides = 5',fontsize=10)


"""
Kmean Neigbors
"""

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap


np.random.seed(5)
classifier1 = KNeighborsClassifier(n_neighbors=5)
print(classifier1)

classifier1.fit(A1, A2)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
                            
x_min=A1[:, 0].min() - 1
x_max=A1[:, 0].max() + 1
y_min=A1[:, 1].min() - 1
y_max=A1[:, 1].max() + 1
h = 1000
#h = .02


"""
https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html
numpy.meshgrid
"""
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

"""
numpy.arange
https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.arange.html
np.arange(3,7)
array([3, 4, 5, 6])
"""

#print(xx.ravel())
#print(yy.shape)
#print((np.c_[xx.ravel(), yy.ravel()]).shape)

"""

classifier1
deberia ser Kmean = KMeans(init='k-means++', n_clusters=5)
"""

# Obtain labels for each point in mesh. Use last trained model.
Z=classifier1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax7=fig.add_subplot(4,2,7)

#Display an image, i.e. data on a 2D regular raster.
"""
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')
"""
#c='green'
plt.scatter(A1[:, 0], A1[:, 1], cmap='viridis',marker=".", s=50)
#plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
#plt.scatter(X[:, 0], X[:, 1], c='r', cmap=cmap_bold,edgecolor='k', s=20)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=500, marker='.')
#plt.scatter(centers[:, 0], centers[:, 1], c='r', cmap=cmap_bold,s=100, marker='o')
ax7.set_xlim((0, 200000000))
ax7.set_ylim((0, 150))

#plt.xlim(xx.min(), xx.max())
#plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i, weights = '%s')", fontsize=10)


"""
Evaluando el mejor valor de K, segun Cross Validation
"""
from sklearn.model_selection import cross_val_score

k_range = range(1, 80)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train,cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
#print (k_scores)

ax8=fig.add_subplot(4,2,8) 
#plt.scatter(B1['Monto Total Solicitud Base'], B1['Cantidad_de_Lineas'], c=y_kmeans, s=10, cmap='plasma')

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN',fontsize=8)
plt.ylabel('Cross-Validated Accuracy',fontsize=8)



plt.tight_layout()
plt.show()



"""
#############################################################
Sirve para ver el score de entrenamiento y de test

from sklearn.neighbors import KNeighborsRegressor
cross_validate(KNeighborsRegressor(),A1,A2,cv=5)
cross_validate(KNeighborsRegressor(n_neighbors=10),A1,A2,cv=5)
#############################################################
Sirve para la validacion de modelos

from sklearn.model_selection import validation_curve
train_scores, test_scores = validation_curve(KNeighborsRegressor(),A1,A2,param_name='n_neighbors',param_range=n,cv=5)
train_scores
np.mean(train_scores,axis=1)
#############################################################
Evaluacion del modelo con cierta cantida de datos, dados en el primer array
from sklearn.model_selection import learning_curve

learning_curve(KNeighborsRegressor(n_neighbors=6),A1,A2,cv=5)
plt.plot(samples,np.mean(train,axis=1))
plt.plot(samples,np.mean(test,axis=1))

Para ver la curva de aprendizaje

#############################################################

"""







"""
from sklearn.model_selection import cross_validate
k_range = range(1, 80)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_validate(knn, X_train, y_train,cv=10, scoring='accuracy')
    k_scores.append(scores.mean())    
"""


"""
Mejoras de graficos

https://www.python-course.eu/matplotlib_multiple_figures.php
https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/gridspec_multicolumn.html#sphx-glr-gallery-subplots-axes-and-figures-gridspec-multicolumn-py
"""

"""                      
Tips=db1['Cantidad_de_Lineas','Unidad_Negocio_num','Semana de creacion','Monto Total Solicitud Base']
tips = sns.load_dataset("tips")
g = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
bins = np.linspace(0, 60, 13)
g.map(plt.hist, "total_bill", color="steelblue", bins=bins)
"""
                             

"""
plt.scatter(A['Monto Total Solicitud Base'], A['Semana de creacion'], s=5, cmap='plasma')
plt.scatter(A['Monto Total Solicitud Base'], A['Unidad_Negocio_num'], s=5, cmap='plasma')
"""


"""
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

C=A[(A['Unidad_Negocio_num']==7)&(A['Tipo presupuesto']=='OPEX')]
type(C)
plt.scatter(C['Monto Total Solicitud Base'], C['Semana de creacion'], s=20, cmap=cmap_bold)


C1= C[['Monto Total Solicitud Base','Semana de creacion']]
kmeans = KMeans(init='k-means++',n_clusters=5, n_init=10)
kmeans.fit(C1)
y_kmeans = kmeans.predict(C1)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=80, alpha=1)
C1=C1.to_numpy() #conversion a Array


                            
x_max, x_min=C1[:, 0].max() + 1, C1[:, 0].min() - 1 #linea Ok para realizar los valores mx y min
y_max, y_min=C1[:, 1].max() + 1, C1[:, 1].min() - 1 #linea Ok para realizar los valores mx y min
h = 200 #Paso de la malla
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h)) #creacion de malla


Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(C['Monto Total Solicitud Base'], C['Semana de creacion'], s=20, cmap=cmap_bold)


from sklearn import neighbors
Neighbors = KNeighborsClassifier(n_neighbors=5)
X = C[['Monto Total Solicitud Base']]
Y= C[['Semana de creacion']]
Neighbors.fit(X,Y)




"""

"""
plt.scatter(C['Monto Total Solicitud Base'], C['Unidad_Negocio_num'], s=5, cmap='plasma')
plt.scatter(C['Cantidad_de_Lineas'], C['Semana de creacion'], s=5, cmap='plasma')

"""
"""
from matplotlib.colors import ListedColormap
from sklearn import neighbors


cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

X=C[['Monto Total Solicitud Base']]
Y=C[['Semana de creacion']]                  
                            
for weights in ['uniform', 'distance']:
    Neighbors = neighbors.KNeighborsClassifier(n_neighbors=5, weights=weights)
    Neighbors.fit(X,Y)
    x_max, x_min=X[:, 0].max() + 1, X[:, 0].min() - 1
    y_max, y_min=X[:, 1].max() + 1, X[:, 1].min() - 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    Z = Neighbors.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], cmap=cmap_bold,s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
plt.show()
"""