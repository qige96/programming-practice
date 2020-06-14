import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score

from ensemble import *

print('\nAdaBoost')
print('---------------------------------------------------------------------')
X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
y_train[y_train==0] = -1
y_test[y_test==0] = -1

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=1)
myada = MyAdaBoostClassifier(tree)
myada.fit(X_train, y_train)
print('My AdaBoost:', accuracy_score(y_test, myada.predict(X_test)))

from sklearn.ensemble import AdaBoostClassifier
skada = AdaBoostClassifier(n_estimators=100, random_state=0)
skada.fit(X_train, y_train)
print('Sk AdaBoost:', accuracy_score(y_test, skada.predict(X_test)))


print('\nRandom Forest')
print('---------------------------------------------------------------------')
myforest = MyRandomForestClassifier()
myforest.fit(X_train, y_train)
# print(myforest.predict(X_test))
print('My forest:', accuracy_score(y_test, myforest.predict(X_test)))

from sklearn.ensemble import RandomForestClassifier
skforest = RandomForestClassifier()
skforest.fit(X_train, y_train)
print('SK forest:', accuracy_score(y_test, skforest.predict(X_test)))