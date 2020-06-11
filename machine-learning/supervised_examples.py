'''
Some use cases for supervised learning algorithms.
'''
import numpy as np
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from supervised import *

print('Linear Regression')
print('---------------------------------------------------------------------')
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print('My MSE:', mean_squared_error(y_test, y_pred))
lr.fit_analitical(X_train, y_train)
y_pred = lr.predict(X_test)
print('My MSE:', mean_squared_error(y_test, y_pred), '(analytical method)')

from sklearn.linear_model import LinearRegression as SKLinearRegression
lr = SKLinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print('Sk MSE:', mean_squared_error(y_test, y_pred))



print('\nLogistic Regression')
print('---------------------------------------------------------------------')
iris = load_iris()
X = iris.data[iris.target!=2]   # only use samples with class 0 or 1
y = iris.target[iris.target!=2] # only use samples with class 0 or 1
X_train, X_test, y_train, y_test = train_test_split(X, y)
lclf = LogisticRegression()
lclf.fit(X_train, y_train)
y_pred = lclf.predict(X_test)
print('My Accuracy:', accuracy_score(y_test, y_pred))

from sklearn.linear_model import LogisticRegression as SKLogisticRegression
lclf = SKLogisticRegression()
lclf.fit(X_train, y_train)
y_pred = lclf.predict(X_test)
print('Sk Accuracy:', accuracy_score(y_test, y_pred))


print('\nSupport Vector Machine')
print('---------------------------------------------------------------------')
lsvm = LinearSVM()
y_train[y_train==0] = -1
y_test[y_test==0] = -1
lsvm.fit(X_train, y_train)
y_pred = lsvm.predict(X_test)
print(lsvm.w, lsvm.b)
print(y_pred, y_test)
print('My Accuracy:', accuracy_score(y_test, y_pred))



print('\nK Nearest Neighbors')
print('---------------------------------------------------------------------')
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)
clf = KNNClissifier()
clf.fit(X_train, y_train, 5)
y_pred = clf.predict(X_test)
print(y_pred, y_test)
print('My Accuracy:', accuracy_score(y_test, y_pred))

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Sk Accuracy:', accuracy_score(y_test, y_pred))
