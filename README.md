# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1)Choose the number of clusters (K): 
Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.

2)Initialize cluster centroids: 
Randomly select K data points from your dataset as the initial centroids of the clusters.

3)Assign data points to clusters: 
Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically  done using Euclidean distance, but other distance metrics can also be used.

4)Update cluster centroids: 
Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.

5)Repeat steps 3 and 4: 
Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally.

6)Evaluate the clustering results: 
Once convergence is reached, evaluate the quality of the clustering results. This can be done using various metrics such as the within-cluster sum of squares (WCSS), silhouette coefficient, or domain-specific evaluation criteria.

7)Select the best clustering solution: 
If the evaluation metrics allow for it, you can compare the results of multiple clustering runs with different K values and select the one that best suits your requirements

## Program:
```
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: SANDEEP S
RegisterNumber:  212223220092
```
```python


import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")
print(data.head())

data.info()

print(data.isnull().sum())

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
  kmeans = KMeans(n_clusters = i, init = "k-means++")
  kmeans.fit(data.iloc[:, 3:])
  wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")

km = KMeans(n_clusters = 5)
km.fit(data.iloc[:, 3:])
plt.show()
y_pred = km.predict(data.iloc[:, 3:])
print(y_pred)

data["cluster"] = y_pred
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c = "red", label = "cluster0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c = "black", label = "cluster1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c = "blue", label = "cluster2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c = "green", label = "cluster3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c = "magenta", label = "cluster4")
plt.legend()
plt.title("Customer Segments")
```
## Output:
### data.head():

![image](https://github.com/user-attachments/assets/81a82339-9c00-4533-943b-ff43b16d74e3)


### data.info():

![image](https://github.com/user-attachments/assets/f85e6a8c-fdeb-462a-bf2e-74c31bf17a47)


### NULL VALUES:

![image](https://github.com/user-attachments/assets/a679f98b-8977-4cdb-bac5-8ebbac500ef5)


### ELBOW GRAPH:

![image](https://github.com/user-attachments/assets/7a0cf98b-c542-414d-af73-9c396d96913a)


### CLUSTER FORMATION:

![image](https://github.com/user-attachments/assets/3441487a-fca3-4ae6-928b-b25d4f129fe7)



### PREDICICTED VALUE:

![image](https://github.com/user-attachments/assets/de09c048-836a-48c2-8c11-522fb0adca14)


### FINAL GRAPH(D/O):

![image](https://github.com/user-attachments/assets/de334ca9-c365-490a-b4b3-d267ce933a09)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
