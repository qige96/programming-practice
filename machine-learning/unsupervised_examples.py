import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from unsupervised import *

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

print('\nK-means clustering')
print('---------------------------------------------------------------------')
mykm = MyKMeans(k=3)
mykm.fit(X_train)
print('My KMeans:')
print(mykm.cluster_centers)
print(mykm.predict(X_test))
print()

from sklearn.cluster import KMeans
skkm = KMeans(n_clusters=3)
skkm.fit(X_train)
print('Sk KMeans:')
print(skkm.cluster_centers_)
print(skkm.predict(X_test))