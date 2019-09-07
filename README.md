# Data Scientist
![1](https://user-images.githubusercontent.com/17385297/50380218-267ddd80-063f-11e9-931b-53066b31db85.PNG)


## Data cleansing
...is the process of detecting and correcting (or removing) corrupt or inaccurate records from a record set, table, or database and refers to identifying incomplete, incorrect, inaccurate or irrelevant parts of the data and then replacing, modifying, or deleting the dirty or coarse data. Data cleansing may be performed interactively with data wrangling tools, or as batch processing through scripting.

## Data wrangling
...the process of manually converting or mapping data from one "raw" form into another format that allows for more convenient consumption of the data with the help of semi-automated tools. This may include further munging, data visualization, data aggregation, training a statistical model, as well as many other potential uses. Data munging as a process typically follows a set of general steps which begin with extracting the data in a raw form from the data source, "munging" the raw data using algorithms (e.g. sorting) or parsing the data into predefined data structures, and finally depositing the resulting content into a data sink for storage and future use. 

![data wrangling 2](https://user-images.githubusercontent.com/17385297/50380033-2af3c780-063a-11e9-81ce-13c3888c313d.PNG)

![data wrangling 3](https://user-images.githubusercontent.com/17385297/50380034-2e874e80-063a-11e9-9cfb-8fd6961472c2.PNG)

![data wrangling](https://user-images.githubusercontent.com/17385297/50380035-33e49900-063a-11e9-9289-9f2de4e7ee9d.PNG)


## Step 2: Exploratory Data Analysis
At a high level, EDA is the practice of using visual and quantitative methods to understand and summarize a dataset without making any assumptions about its contents. It is a crucial step to take before diving into machine learning or statistical modeling because it provides the context needed to develop an appropriate model for the problem at hand and to correctly interpret its results.

## Step 3: Dealing with Missing Values

![dealing with missing values](https://user-images.githubusercontent.com/17385297/50379995-47433480-0639-11e9-82ca-2d10cb5de534.PNG)

## Step 4: Dealing with Outliers

This is not a tutorial on drafting a strategy to deal with outliers in your data when modeling; there are times when including outliers in modeling is appropriate, and there are times when they are not (regardless of what anyone tries to tell you). This is situation-dependent, and no one can make sweeping assertions as to whether your situation belongs in column A or column B. 

## Step 5: Dealing with Imbalanced Data


## Step 6: Data Transformations
In statistics, data transformation is the application of a deterministic mathematical function to each point in a data set — that is, each data point zi is replaced with the transformed value yi = f(zi), where f is a function. Transforms are usually applied so that the data appear to more closely meet the assumptions of a statistical inference procedure that is to be applied, or to improve the interpretability or appearance of graphs. 


## Step 7: Finishing Touches & Moving Ahead

Source: https://www.kdnuggets.com/2017/06/7-steps-mastering-data-preparation-python.html


## Step 8: Pending
Source: https://www.datacamp.com/community/tutorials/understanding-recursive-functions-python<br/>
Source: https://www.datacamp.com/community/tutorials/decision-tree-classification-python<br/>
Source: https://www.kdnuggets.com/2018/04/key-algorithms-statistical-models-aspiring-data-scientists.html<br/>
Source: https://www.quantinsti.com/blog/top-10-machine-learning-algorithms-beginners?utm_campaign=News&utm_medium=Community&utm_source=DataCamp.com<br/>


https://datatofish.com/upgrade-pip/

## Actualizacion de PIP
C:\Python27>python.exe -m pip install --upgrade pip <br/>
C:\Python27\Scripts>pip --version <br/>
C:\Python27\Scripts>pip install --upgrade pandas <br/>
C:\Python27\Scripts>pip install --upgrade scikit-learn <br/>
C:\Python27\Scripts>pip install --upgrade seaborn <br/>
C:\Python27\Scripts>pip install --upgrade matplotlib <br/>
C:\Python27\Scripts>pip install --upgrade numpy <br/>



[7 Steps to Mastering Basic Machine Learning with Python](https://www.kdnuggets.com/2019/01/7-steps-mastering-basic-machine-learning-python.html)



## Bokeh

conda install bokeh
pip install bokeh

https://bokeh.pydata.org/en/latest/docs/user_guide/quickstart.html#installation

```python
from bokeh.plotting import figure, output_file, show
output_file("test.html")
p = figure()
p.line([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=2)
show(p)
```


## Panel
https://medium.com/@philipp.jfr/panel-announcement-2107c2b15f52
https://panel.pyviz.org/
https://github.com/pyviz/panel

```
conda install -c pyviz panel 
pip install panel.
```
## Pyviz

http://pyviz.org/


## Classification

Classification is one of the main methods of supervised learning, and the manner in which prediction is carried out as relates to data with class labels. Classification involves finding a model which describes data classes, which can then be used to classify instances of unknown data. The concept of training data versus testing data is of integral importance to classification. Popular classification algorithms for model building, and manners of presenting classifier models, include (but are not limited to) decision trees, logistic regression, support vector machines, and neural networks.

### Regression

Regression is similar to classification, in that it is another dominant form of supervised learning and is useful for predictive analysis. They differ in that classification is used for predictions of data with distinct finite classes, while regression is used for predicting continuous numeric data. As a form of supervised learning, training/testing data is an important concept in regression as well.

### Clustering
 
Clustering is used for analyzing data which does not include pre-labeled classes. Data instances are grouped together using the concept of maximizing intraclass similarity and minimizing the similarity between differing classes. This translates to the clustering algorithm identifying and grouping instances which are very similar, as opposed to ungrouped instances which are much less-similar to one another. As clustering does not require the pre-labeling of classes, it is a form of unsupervised learning.

https://www.kdnuggets.com/2019/01/7-steps-mastering-basic-machine-learning-python.html
https://towardsdatascience.com/an-introduction-to-clustering-algorithms-in-python-123438574097


## Kmeans
https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1

“the objective of K-means is simple: group similar data points together and discover underlying patterns. To achieve this objective, K-means looks for a fixed number (k) of clusters in a dataset.”

A cluster refers to a collection of data points aggregated together because of certain similarities.

You’ll define a target number k, which refers to the number of centroids you need in the dataset. A centroid is the imaginary or real location representing the center of the cluster.

Every data point is allocated to each of the clusters through reducing the in-cluster sum of squares.

In other words, the K-means algorithm identifies k number of centroids, and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible.

The ‘means’ in the K-means refers to averaging of the data; that is, finding the centroid.

```python
from sklearn.cluster import KMeans
Kmean = KMeans(n_clusters=2)
Kmean.fit(X)
```

Here is the code for finding the center of the clusters:

```python
Kmean.cluster_centers_
```

Here is the code for getting the labels property of the K-means clustering example dataset; that is, how the data points are categorized into the two clusters.

```python
Kmean.labels_
```

For example, let’s use the code below for predicting the cluster of a data point:

```python
sample_test=np.array([-3.0,-3.0])
second_test=sample_test.reshape(1, -1)
Kmean.predict(second_test)
```

## Total Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
%matplotlib inline
X= -2 * np.random.rand(100,2)
X1 = 1 + 2 * np.random.rand(50,2)
X[50:100, :] = X1
plt.scatter(X[ : , 0], X[ :, 1], s = 50, c = ‘b’)
plt.show()
from sklearn.cluster import KMeans
Kmean = KMeans(n_clusters=2)
Kmean.fit(X)
Kmean.cluster_centers_
plt.scatter(X[ : , 0], X[ : , 1], s =50, c=’b’)
plt.scatter(-0.94665068, -0.97138368, s=200, c=’g’, marker=’s’)
plt.scatter(2.01559419, 2.02597093, s=200, c=’r’, marker=’s’)
plt.show()
Kmean.labels_
sample_test=np.array([-3.0,-3.0])
second_test=sample_test.reshape(1, -1)
Kmean.predict(second_test)
```
