'''
Some use cases for supervised learning algorithms.
'''
import numpy as np
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from supervised import *


# ============================================================================
#                                Regression
# ============================================================================

print('Linear Regression')
print('---------------------------------------------------------------------')
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target)
lr = MyLinearRegression()
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
lclf = MyLogisticRegression()
lclf.fit(X_train, y_train)
y_pred = lclf.predict(X_test)
print('My Accuracy:', accuracy_score(y_test, y_pred))

from sklearn.linear_model import LogisticRegression as SKLogisticRegression
lclf = SKLogisticRegression()
lclf.fit(X_train, y_train)
y_pred = lclf.predict(X_test)
print('Sk Accuracy:', accuracy_score(y_test, y_pred))

# ============================================================================
#                               Decision Tree
# ============================================================================

print('\nDecision Tree')
print('---------------------------------------------------------------------')
xigua2 = np.array([
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
        # ----------------------------------------------------
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
    ])
mytree = MyDecisionTreeClassifier()
X = xigua2[:, :-1]
y = xigua2[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
mytree.fit(X_train, y_train)
print('predicted:', mytree.predict(X_test), 'true:', y_test)
print('')
mytree.print_tree()



# ============================================================================
#                           Support Vector Machine
# ============================================================================


print('\nSupport Vector Machine')
print('---------------------------------------------------------------------')
mysvm = MyLinearSVM()
y_train[y_train==0] = -1
y_test[y_test==0] = -1
mysvm.fit(X_train, y_train)
y_pred = mysvm.predict(X_test)
print(mysvm.w, mysvm.b)

print('My Accuracy:', accuracy_score(mysvm.predict(X_test), y_test))

from sklearn.svm import LinearSVC
sksvm = LinearSVC()
sksvm.fit(X_train, y_train)
print(sksvm.coef_, sksvm.intercept_) 
print('Sk Accuracy:', accuracy_score(sksvm.predict(X_test), y_test))



# ============================================================================
#                                Naive Bayes
# ============================================================================


print('\nDescrete Naive Bayes')
print('---------------------------------------------------------------------')
xigua2 = np.array([
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
        # ----------------------------------------------------
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
    ])
from sklearn.preprocessing import LabelEncoder
for i in range(xigua2.shape[1]):
    le = LabelEncoder()
    xigua2[:,i] = le.fit_transform(xigua2[:,i])
xigua2 = xigua2.astype(np.int).repeat(20, axis=0)
X = xigua2[:, :-1]
y = xigua2[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y)

mynnb = MyCategoricalNBC()
mynnb.fit(X_train, y_train)
print('My Accuracy:', accuracy_score(mynnb.predict(X_test), y_test))

from sklearn.naive_bayes import CategoricalNB 
sknnb = CategoricalNB()
sknnb.fit(X_train, y_train)
print('SK Accuracy:', accuracy_score(sknnb.predict(X_test), y_test))


print('\nContinuous Naive Bayes')
print('---------------------------------------------------------------------')
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

mycnb = MyGaussianNBC()
mycnb.fit(X_train, y_train)
print('My Accuracy:', accuracy_score(mycnb.predict(X_test), y_test))

from sklearn.naive_bayes import GaussianNB
skcnb = GaussianNB()
skcnb.fit(X_train, y_train)
print('Sk Accuracy:', accuracy_score(skcnb.predict(X_test), y_test))

# ============================================================================
#                            K Nearest Neighbors
# ============================================================================

print('\nK Nearest Neighbors')
print('---------------------------------------------------------------------')
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)
kclf = MyKNNClissifier()
kclf.fit(X_train, y_train, 2)
y_pred = kclf.predict(X_test)
print(y_pred, y_test)
print('My Accuracy:', accuracy_score(y_test, kclf.predict(X_test)))

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Sk Accuracy:', accuracy_score(y_test, clf.predict(X_test)))

xigua2 = np.array([
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
        # ----------------------------------------------------
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
    ])

X = xigua2[:, :-1]
y = xigua2[:, -1]