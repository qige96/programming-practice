'''
Simple implementation of some supervised learning algorithms.
'''
import numpy as np

from utils import binarize, sigmoid

# ============================================================================
#                                Regression
# ============================================================================


class LinearRegression:

    def __init__(self):
        self.w = None

    def _loss_function(self, X, y, w):
        '''mean square as loss function'''
        y_hat = X.dot(w)
        return np.sum((y - y_hat)**2)

    def _gradient_descent(self, X, y, w, eta):
        '''compute new weights by batch gradient descent'''
        y_hat = X.dot(w)
        diff = y - y_hat
        grad = eta * (y - y_hat).reshape([-1,1]) * X * 1.0/len(X)
        w += grad.sum(axis=0)
        return w

    def fit_analitical(self, X, y):
        '''
        Train the logistic regression model, by analytical method

        Parameters
        ----------
        X: ndarray of shape (m, n)
            sample data where row represent sample and column represent feature
        y: ndarray of shape (m,)
            labels of sample data

        Returns
        -------
        self
            trained model
        '''
        X, y = np.array(X), np.array(y)
        X = np.hstack([X, np.ones([len(X),1])]) # add bias
        w = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
        self.b = w[-1]
        self.w = w[:-1]
        return self


    def fit(self, X, y, eta=1e-8, epslon=1e-6, max_iter=100000):
        '''
        Train the simple linear regression model, by gradient descent

        Parameters
        ----------
        X: ndarray of shape (m, n)
            sample data where row represent sample and column represent feature
        y: ndarray of shape (m,)
            labels of sample data
        eta: float
            learning rate
        epslon: float
            terminate when prev_loss - this_loss < epslon
        max_iter: int
            terminate when number of iterations excceed max_iter

        Returns
        -------
        self
            trained model
        '''
        X = np.hstack([X, np.ones([len(X),1])]) # add bias
        # w = np.random.random(X.shape[1]) 
        w = np.ones(X.shape[1])  # standard practice is to randomly initialize
        prev_loss = self._loss_function(X, y, w)
        for i in range(max_iter):
            w = self._gradient_descent(X, y, w, eta)
            this_loss =  self._loss_function(X, y, w)
            if prev_loss - this_loss < epslon:
                break
            prev_loss = this_loss
        self.b = w[-1]
        self.w = w[:-1]
        return self
    
    def predict(self, X):
        '''
        Make prediction by the trained model.

        Parameters
        ----------
        X: ndarray of shape (m, n)
            data to be predicted, the same shape as trainning data

        Returns
        -------
        C: ndarray of shape (m,)
            Predicted value per sample.
        '''
        return X.dot(self.w) + self.b


class LogisticRegression:

    def __init__(self):
        self.w = None
        self.b = None
    
    def _loss_function(self, X, y, w): 
        '''cross entropy as loss function'''
        y_hat = sigmoid(X.dot(w))
        ll = np.sum(y * np.log(y_hat) - (1-y) * np.log(1-y_hat))
        return ll

    def _gradient_descent(self, X, y, w, eta):
        '''compute new weights by batch gradient descent'''
        y_hat = sigmoid(X.dot(w))
        diff = y - y_hat
        grad = eta * diff.reshape([-1,1]) * X * 1.0/len(X)
        w += grad.sum(axis=0)
        return w

    def fit(self, X, y, eta=1e-5, epslon=1e-6, max_iter=100000):
        '''
        Train the logistic regression classifier model

        Parameters
        ----------
        X: ndarray of shape (m, n)
            sample data where row represent sample and column represent feature
        y: ndarray of shape (m,)
            labels of sample data
        eta: float
            learning rate
        epslon: float
            terminate when prev_loss - this_loss < epslon
        max_iter: int
            terminate when number of iterations excceed max_iter

        Returns
        -------
        self
            trained model
        '''
        X = np.hstack([X, np.ones([len(X),1])]) # add bias
        # standard practice is to randomly initialize
        w = np.ones(X.shape[1])  
        prev_loss =  self._loss_function(X, y, w)
        for i in range(max_iter):
            self.w = self._gradient_descent(X, y, w, eta)
            this_loss = self._loss_function(X, y, w)
            if prev_loss - this_loss < epslon:
                break
            prev_loss = this_loss
        self.b = w[-1]
        self.w = w[:-1]
        return self
    
    def predict(self, X):
        '''
        Make prediction by the trained model.

        Parameters
        ----------
        X: ndarray of shape (m, n)
            data to be predicted, the same shape as trainning data

        Returns
        -------
        C: ndarray of shape (m,)
            Predicted class label per sample.
        '''
        if self.w is None:
            raise Exception("Model haven't been trained!")
        # X = np.hstack([X, np.ones([len(X), 1])])
        return binarize(sigmoid(X.dot(self.w)+self.b))


# ============================================================================
#                               Decision Tree
# ============================================================================

class DecisionTree:

    def __init__(self):
        self.w = None

    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pass


# ============================================================================
#                           Support Vector Machine
# ============================================================================


class LinearSVM:

    def __init__(self):
        self.w = None
        self.b = None

    def _identify_support_vectors(self, X, y):
        pass 

    def _compute_svm_weights(self, X_supp, y_supp):
        '''
        compute weights of a linear SVM with identified support vectors

        X_supp: 2d array - each row represents a support vector
        y_supp: 1d array - label of each support vector, either 1 or -1
        '''
        A = X_supp.dot(X_supp.T)*y_supp
        A = np.vstack([A, y_supp])
        A = np.hstack([A, np.array([1]*len(X_supp)+[0]).reshape([-1,1])])
        b = np.concatenate([y_supp, [0]])
        print(A)
        print(A.shape)
        res = np.linalg.inv(A).dot(b)
        lambdas, w0 = res[:-1], res[-1]
        w = np.sum([lambdas[i]*X_supp[i]*y_supp[i] for i in range(len(X_supp))], axis=0)

        # print('euqation atrix:', A)
        # print('equation vector:', b)
        # print('lambdas:',lambdas, ', w0:', w0)
        return (w, w0)
    
    def fit(self, X, y):
        self.w, self.b = self._compute_svm_weights(X, y)
    
    def predict(self, X):
        print(X.dot(self.w)+self.b)
        return np.where(X.dot(self.w)+self.b<0, -1, 1)

# ============================================================================
#                                Naive Bayes
# ============================================================================

class NaiveBayes:

    def __init__(self):
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, X):
        '''
        Make prediction by the trained model.

        Parameters
        ----------
        X: ndarray of shape (m, n)
            data to be predicted, the same shape as trainning data

        Returns
        -------
        C: ndarray of shape (m,)
            Predicted class label per sample.
        '''
        if self.w == None:
            raise Exception("Model haven't been trained!")
        X = np.hstack([X, np.ones([len(X), 1])])
        return binarize(sigmoid(X.dot(self.w)))


# ============================================================================
#                            K Nearest Neighbors
# ============================================================================


class KNNClissifier:

    def __init__(self):
        self.X = None
        self.y = None
        self.k = None
        self.func_dist = None

    def _top_k_nn(self, centroid, surroundings):
        '''
        return indices of the top k nearest neighbors of centroid 
        from surroundings
        '''
        distances = []
        for vec in surroundings:
            distances.append(self.func_dist(centroid - vec))
        distances = np.array(distances)
        top_k = np.argsort(distances)[:self.k]

        # print('neightbors and distances:', \
        #    [(v, d) for v, d in zip(surroundinds, distances)])
        return top_k

    def fit(self, X, y, k=1, func_dist=None):
        '''
        Train the K nearest neighbor classifier model

        Parameters
        ----------
        X: ndarray of shape (m, n)
            sample data where row represent sample and column represent feature
        y: ndarray of shape (m,)
            labels of sample data
        k: int
            number of neightbors selected to compare
        func_dist: function
            distance function, by default Euclidean distance

        Returns
        -------
        self
            trained model
        '''
        self.X = X
        self.y = y
        self.k = k
        if func_dist == None:
            self.func_dist = np.linalg.norm # Euclidean distance
        else:
            self.func_dist = func_dist

    
    def predict(self, X):
        '''
        Make prediction by the trained model.

        Parameters
        ----------
        X: ndarray of shape (m, n)
            data to be predicted, the same shape as trainning data

        Returns
        -------
        C: ndarray of shape (m,)
            Predicted class label per sample.
        '''
        if self.X is None:
            raise Exception("Model haven't been trained!")

        from collections import Counter
        C = []
        for x in X:
            top_k_indices = self._top_k_nn(x, self.X)
            top_k_labels = self.y[top_k_indices]

            # top_k = self.X[top_k_indices]
            # print('top k neightbors:', top_k)
            # print('class of top k neightbors:', top_k_labels)
            C.append(Counter(top_k_labels).most_common(1)[0][0])

        return np.array(C)

# ============================================================================
#                               Miscellaneous
# ============================================================================


if __name__ == '__main__':
    import doctest
    doctest.testmod()
