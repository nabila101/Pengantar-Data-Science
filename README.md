# Classification
## Import library
```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
%matplotlib inline
```

## Data resource https://archive.ics.uci.edu/ml/datasets/IrisInformasi 

## read dataset
```
data = pd.read_csv('/content/iris.data')
data.head(5)
```
### output 
![image](https://user-images.githubusercontent.com/72323527/232316334-3a762591-5dcf-4da8-ac8e-14992b645136.png)

### atribut data
```
data.columns =["sepal length", "sepal width","petal length", "petal width", "class"]
data.head(5)
```
output
![image](https://user-images.githubusercontent.com/72323527/232316402-b6634184-695f-47d1-a29f-867102d98bc2.png)

### split dataset 
```
from sklearn.model_selection import train_test_split
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print('Jumlah data train:',len(X_train))
print('Jumlah data test :',len(X_test))
```
### output
![image](https://user-images.githubusercontent.com/72323527/232316490-6e37c4ea-98c2-4242-a684-154521eef9a7.png)

## Visualisasi Data
```
import seaborn as sn
import matplotlib.pyplot as plt

## Set theme ##
sn.set_theme(style='darkgrid')
```
```
## Visualisasi Data X ##
data.iloc[:,:-1].hist(figsize=(10,10), color='#01b1b5')
plt.show()
```

### output
![image](https://user-images.githubusercontent.com/72323527/232316600-d1000b15-0383-4ef1-a995-eb953bd9ce5e.png)

![image](https://user-images.githubusercontent.com/72323527/232316653-0af17094-d0b5-444c-97b0-48e65e06a8af.png)

## K-Nearest Neighbor
```
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
```
```
knn_method = KNeighborsClassifier()
knn_method.fit(X_train, y_train) 
y_pred_knn = knn_method.predict(X_test)

score_knn = metrics.accuracy_score(y_pred_knn,y_test)



print('Akurasi metode KNN: ',round(score_knn,8), ' atau ', round(score_knn*100,2),'%', sep='')
```
### output
![image](https://user-images.githubusercontent.com/72323527/232317212-bfcc66c7-e2b9-4ddb-bb0b-a80c0665e99e.png)

```
## Evaluasi model ##
k = 10
mean_acc = []
std_acc = []

for i in range(1,k):
    knn_method = KNeighborsClassifier(n_neighbors = i)
    knn_method.fit(X_train, y_train)
    y_pred_knn = knn_method.predict(X_test)
    mean_acc.append(metrics.accuracy_score(y_pred_knn, y_test))
    std_acc.append(np.std(y_pred_knn==y_test)/np.sqrt(y_pred_knn.shape[0]))
    
mean_acc
```

### output
![image](https://user-images.githubusercontent.com/72323527/232317330-f77841b7-2d57-4336-b64a-9575c4cd7a64.png)

```
## K-NN ##
plt.figure(figsize = (10,5))
plt.plot(range(1,k),mean_acc, color='#01b1b5')
plt.fill_between(range(1,k),
                np.array(mean_acc) - 1 * np.array(std_acc),
                mean_acc + 1 * np.array(std_acc), 
                alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Nilai K')
plt.tight_layout()
plt.show()
```
### output
![image](https://user-images.githubusercontent.com/72323527/232317557-d016024a-5d5f-43b4-8b5e-cae946088109.png)

```
print("Nilai Akurasi terbaik ada pada ", 
      round(np.array(mean_acc).max()*100,2), 
      "% dengan k = ", np.array(mean_acc).argmax()+1, sep='') 
```
### output
![image](https://user-images.githubusercontent.com/72323527/232317820-d8cbe110-cf54-4362-a398-cfe14e7203cb.png)

## Decision Tree

```
from sklearn.tree import DecisionTreeClassifier
tree_method = DecisionTreeClassifier()
tree_method.fit(X_train, y_train)
y_pred_tree = tree_method.predict(X_test)
score_tree = metrics.accuracy_score(y_pred_tree, y_test)
print('Akurasi Prediksi method Tree : ',round(score_tree,8), ' atau ',round(score_tree*100,2),'%', sep='')
```
### output
![image](https://user-images.githubusercontent.com/72323527/232318242-efd91b61-4ff7-4388-9eff-3bce9e0c57e3.png)

```
from sklearn import tree
plt.figure(figsize=(20,20))
tree.plot_tree(tree_method)
plt.show()
```

### output
![image](https://user-images.githubusercontent.com/72323527/232318477-8d714878-38dc-4ed0-8cf0-4f14d9baf3c4.png)
![image](https://user-images.githubusercontent.com/72323527/232318516-671ef735-0505-4cce-9fce-bcd73d07d0fb.png)

## Naive Bayes
```
bayas_method = naive_bayes.BernoulliNB()
bayas_method.fit(X_train, y_train)
y_pred_bayes = bayas_method.predict(X_test)
score_bayes = metrics.accuracy_score(y_pred_bayes, y_test)
print('Akurasi Prediksi Bayes : ',round(score_bayes,8), ' atau ',round(score_bayes*100,2),'%', sep='')
```
### output
![image](https://user-images.githubusercontent.com/72323527/232318704-baf6bb4e-327c-43cb-ab69-3c20790844b2.png)

```
def get_conf_matriks(y_actual, y_predic, cmap = None, title='Tidak ada', ):
    confusion_matrix = metrics.confusion_matrix(y_actual, y_predic)

    
    conf_matrix_value = (confusion_matrix[0,0] + confusion_matrix[1,1]) / sum(sum(confusion_matrix))
    print("Hasil Confution Matriks :" , round(conf_matrix_value*100,2), "%", sep='')
```
```
get_conf_matriks(y_test, y_pred_knn, title='Matriks Confution K-NN')
```
### output
![image](https://user-images.githubusercontent.com/72323527/232318783-7749dfbb-22bd-4f0a-8490-9967ec4dc41e.png)

```
get_conf_matriks(y_test, y_pred_bayes, title='Matriks Confution Nive Bayes')
```
### output
![image](https://user-images.githubusercontent.com/72323527/232318807-861e6c38-4f90-49e2-adba-5f89184b5db0.png)

```
get_conf_matriks(y_test, y_pred_tree, title='Matriks Confution Decition Tree')
```
### output
![image](https://user-images.githubusercontent.com/72323527/232318832-1892aad1-d254-4a99-b1bc-eea791e1e1ef.png)

## Kesimpulan 
### Dari ketiga metode yang telah dilakukan yaitu  K-NN, Decitioin Tree, dan Nive Bayes, untuk nilai confution matriks dari ketiga metode ini sama yaitu 26.32%.
### Namun untuk prediksi Bayes memiliki hasil yang berbeda yaitu 0.28947368 atau 28.95%. Prediksi ini berbeda dari dua metode lainnya yang memiliki hasil yang sama yaitu 0.97368421 atau 97.37%

# Regression

## Read data
```
#read data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sn
## Set theme ##
sn.set_theme(style='darkgrid')
data_olah = pd.read_csv('/content/Real-estate-valuation-data-set.csv')
```
## data sources  https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set

```
data_olah.head(5)
```
### output
![image](https://user-images.githubusercontent.com/72323527/232319486-e5f86a4c-66b8-446b-a7cb-10d15f38528a.png)

```
data_olah.columns
```
### output
![image](https://user-images.githubusercontent.com/72323527/232319497-7ec0672f-b902-497d-acb6-329426346d49.png)

## Visualisasi Data
```
data_olah.iloc[:,:-1].hist(figsize=(10,10), color='#01b1b5')
plt.show()
```
### output
![image](https://user-images.githubusercontent.com/72323527/232319555-c0a363e9-f9e7-4c97-a06e-0ed89eb2d706.png)

```
x = data_olah[['No', 'X1 transaction date', 'X2 house age',
       'X3 distance to the nearest MRT station',
       'X4 number of convenience stores', 'X5 latitude', 'X6 longitude',
       'Y house price of unit area']]
y = data_olah[['Y house price of unit area']]
```
```
data_olah = data_olah.drop("No", axis=1)
```
## Split data
```
#split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
```
```
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
print(model.coef_)
```
### output
![image](https://user-images.githubusercontent.com/72323527/232319599-738ac3bf-651c-4848-b8c8-2bf8d12a378b.png)

```
print(model.intercept_)
```
### output
![image](https://user-images.githubusercontent.com/72323527/232319619-c3995063-5e2e-4678-a125-42ecb350807b.png)

```
predictions = model.predict(x_test)
plt.scatter(y_test, predictions)
```
### output
![image](https://user-images.githubusercontent.com/72323527/232319658-eda8cd7e-d088-4bbf-87d2-45a34e35f8bb.png)

```
plt.hist(y_test - predictions)
```
### output

![image](https://user-images.githubusercontent.com/72323527/232319692-3565bf68-f100-4f97-8bf7-35fbe25d03cf.png)

```
from sklearn import metrics
metrics.mean_absolute_error(y_test, predictions)
```
### output 
![image](https://user-images.githubusercontent.com/72323527/232319731-c817e075-2561-413f-af8f-b878222b8fbb.png)

```
metrics.mean_absolute_error(y_test, predictions)
```
### output
![image](https://user-images.githubusercontent.com/72323527/232319802-f9b42aad-606b-4dda-a559-aaa96aa353b9.png)

```
np.sqrt(metrics.mean_squared_error(y_test, predictions))
```
### output
![image](https://user-images.githubusercontent.com/72323527/232319812-828a2473-32e2-4994-9583-c3ff979c4fab.png)


## Kesimpulan
### Hasil MSE yang didapatkan menggunakan metode Regresi untuk dataset ini adalah 8.748202162678353e-14

# Clustering
## Import library
```
import pandas as pd
import numpy as np
```
## Data sources https://archive.ics.uci.edu/ml/datasets/BuddyMove+Data+Set
## Read data
```
dataolah = pd.read_csv('/content/buddymove_holidayiq.csv')
dataolah.head() ## .tail()
```
### output
![image](https://user-images.githubusercontent.com/72323527/232320054-15a9e0bb-ba0a-4fd0-bd00-e5a622fe73ed.png)

```
X = dataolah.iloc[:, [3,4]].values
```
```
from sklearn.cluster import KMeans

import seaborn as sns
import matplotlib.pyplot as plt
dataolah.iloc[:,:-1].hist(figsize=(10,10), color='#01b1b5')
plt.show()
```
### output
![image](https://user-images.githubusercontent.com/72323527/232320139-b6885a51-54c6-47a0-92ae-6750394e013f.png)
![image](https://user-images.githubusercontent.com/72323527/232320151-6cf815ea-44e7-4818-b6c7-33eb2cd4b18c.png)

```
## mencari kelompok terbaik ##
wcss = []
for i in range(1,11):
    method_kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    method_kmeans.fit(X)
    wcss.append(method_kmeans.inertia_)
```
### output
![image](https://user-images.githubusercontent.com/72323527/232320223-3e8dcb2f-75d5-4a48-a86a-5439e02b9ccc.png)

```
sns.set_theme(style='darkgrid')
```
```
plt.plot(range(1,11), wcss)
plt.title('Pemilihan k claster')
plt.xlabel('Cluster Number')
plt.ylabel('WCSS')
plt.show()
```
### output
![image](https://user-images.githubusercontent.com/72323527/232320257-09ebf7d3-40eb-4a3e-ae98-75a32885b011.png)

```
method_kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = method_kmeans.fit_predict(X)
y_kmeans
```
### output
![image](https://user-images.githubusercontent.com/72323527/232320285-84b62754-4d52-4465-91ab-1993a5929a52.png)

```
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = '#008080', label = 'Sports')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = '#FF7744', label = 'Religious')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 50, c = '#FF8080', label = 'Nature')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 50, c = '#A155B9', label = 'Theather')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 50, c = '#00639b', label = 'Shopping')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 50, c = '#00639b', label = 'Picnic')
plt.scatter(method_kmeans.cluster_centers_[:, 0], method_kmeans.cluster_centers_[:, 1], s = 100, c = '#555555', label = 'Centroids')
plt.title('Reviews Cluster')
plt.xlabel('Numbers of Reviews')
plt.ylabel('Number of Reviews (1-100)')
plt.legend()
plt.show()
```
### output
![image](https://user-images.githubusercontent.com/72323527/232320309-7573a766-3560-439b-802a-2b1300a95a3d.png)


