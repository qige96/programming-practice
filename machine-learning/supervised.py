'''
Simple implementation of some supervised learning algorithms.
'''
import numpy as np

from utils import binarize, sigmoid

# ============================================================================
#                                Regression
# ============================================================================

class MyLinearRegression:

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


class MyLogisticRegression:

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


class DTreeNode:
    def __init__(self, attr_id, X, y, children, label):
        self.attr_id = attr_id 
        self.X = X
        self.y = y
        self.children = children
        
        self.label = label # only make sense when node is a leaf
    
    def __str__(self):
        return '<attr_id: {0}, label: {1}, y: {2}>'.format(self.attr_id, self.label, self.y)
    
    def __repr__(self):
        return self.__str__()
    
    def is_leaf(self):
        return self.children is None
    
    def print_all(self):
        print('attr_id:', self.attr_id)
        print('X:', self.X)
        print('y:', self.y)
        print('children:', self.children)
        print('label:', self.label)


class MyDecisionTreeClassifier:
    '''
    Warning: this model a product of code practice,
    it cannot handle continuous value, nor could it handle
    missing value. This tree requires value in test samples 
    all included in training samples.
    '''
    def __init__(self):
        self.X = None
        self.y = None
        self.tree = None
        def _gini(self, branch:list):
        '''compute Gini index of a branch of a tree split'''
        return 1 - sum([(d/sum(branch))**2 for d in branch])

    def _gini_index(self, split):
        '''compute Gini index of a tree split'''
        num_D = sum([sum(branch) for branch in split])
        return sum([sum(branch)/num_D * self._gini(branch) for branch in split])
    
    def _count_samples_in_split(self, split):
        '''
        count instances in each branches of a split
        
        Examples
        --------
        >>> spl = [{'Data': np.array([['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑'],
        ...          ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑'],
        ...          ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘'],
        ...          ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑'],
        ...          ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑'],
        ...          ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘']]),
        ...   'labels': np.array(['好瓜', '好瓜', '好瓜', '好瓜', '坏瓜', '坏瓜'])},
        ...  {'Data': np.array([['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑'],
        ...          ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑'],
        ...          ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘'],
        ...          ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑'],
        ...          ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑']]),
        ...   'labels': np.array(['好瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜'])}]
        >>> count_samples_in_split(split)
        [[4, 2], [1, 4]]
        '''
        split_of_numbers = []
        classes = set(self.y)
        for b in split:
            split_of_numbers.append([
                len(b['Data'][b['labels']==c]) for c in classes
            ])
        return split_of_numbers

    def _one_split(self, Data, labels, attr_id):
        '''split a a set of samples by biven attribute'''
        attr_vals = set(Data[:, attr_id])
        split = []
        for v in attr_vals:
            split.append({
                'Data': Data[Data[:, attr_id]==v], 
                'labels': labels[Data[:, attr_id]==v]
                })
        return split

    def _split(self, Data, labels):
        '''
        try all attributes to partition a set of samples 
        and find the best split with lowest impurity
        '''
        partition = []     # all possible splits
        gini_indices = []  # gini indices of each possible split

        for attr_id in range(Data.shape[1]):
            one_split = self._one_split(Data, labels, attr_id)
            partition.append(one_split)
            gini_indices.append(self._gini_index(
                self._count_samples_in_split(one_split)))

        attr_id = np.argmin(gini_indices)  # attribute that produce best split
        return attr_id, partition[attr_id]

    
    def _build_tree(self, Data, labels):
        '''recursively build a decision tree'''
        if len(set(labels)) == 1:
            # all instances belong to one class, make a leaf node
            return DTreeNode(None, Data, labels, None, labels[0])

        attr_id, split = self._split(Data, labels)
        children = {}
        for branch in split:
            attr_val = branch['Data'][:, attr_id][0]
            # build a sub-tree given a subset of data
            children[attr_val] = self._build_tree(branch['Data'], branch['labels'])

        return DTreeNode(attr_id, Data, labels, children, None)

    def _print_tree(self, node, depth):
        '''recursively preint the decision tree'''
        for child_key, child_val in node.children.items():
            print("|      " * depth, child_key , "+---", child_val)
            if not child_val.is_leaf():
                self._print_tree(child_val, depth+1)
        
    def print_tree(self):
        '''preint the decision tree structure'''
        if self.tree is None:
            raise Exception("Model hasn't been trained")
        print(self.tree)
        self._print_tree(self.tree, 0)

    def fit(self, X, y):
        '''
        Train a decision tree classifier model

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
        self.X = X,
        self.y = y
        self.tree = self._build_tree(X, y)
        return self
        
    def _predict(self, x):
        '''recursively traverse the tree and find and label for sample x'''
        node = self.tree
        while node.children is not None:
            node = node.children.get(x[node.attr_id])
        return node.label
    
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
        if self.tree is None:
            raise Exception("Model hasn't been trained")
        assert len(X.shape)==2, 'Input X must be a 2d array'
        results = []
        for x in X:
            results.append(self._predict(x))
        return np.array(results)


# ============================================================================
#                           Support Vector Machine
# ============================================================================


class MyLinearSVM:

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

class MyCategoricalNBC:
    '''nominal naive bayes classifier'''

    def __init__(self):
        self.X = None
        self.y = None

    def fit(self, X, y):
        '''
        Train the nominal naive bayes classifier model

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
        self.X = X
        self.y = y
        return self
    
    def _predict(self, x):
        '''
        compute probabilities and make prediction.
        '''
        probas = {}
        clss = list(set(self.y))

        # compute probability for each attributes in x
        for c in clss:
            probas[c] = []
            dat = self.X[self.y==c]
            for attr_id, attr_val in enumerate(x):
                count = 0
                for row in dat:
                    if attr_val == row[attr_id]:
                        count += 1
                # use laplace smoothing
                probas[c].append((count+1)/(len(dat)+len(x)))
            probas[c].append(len(dat)/len(self.X))

        final_probas = {}
        from functools import reduce
        for c, prbs in probas.items():
            # theoretically not final probability because not divided by Pr(x)
           final_probas[reduce(lambda x,y:x * y, prbs)] = c
        return final_probas[max(final_probas)]
    
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
        return np.array([self._predict(x) for x in X])


class MyGaussianNBC:
    '''Gaussian continuous naive bayes classifier'''

    def __init__(self):
        self.X = None
        self.y = None

    def fit(self, X, y):
        '''
        Train the Gaussian continuous naive bayes classifier model

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
        self.X = X
        self.y = y
        return self

    def _predict(self, x):
        '''compute probabilities and make prediction.'''
        from scipy.stats import norm
        probas = {}
        clss = list(set(self.y))

        # compute probability for each attributes in x
        for c in clss:
            probas[c] = []
            dat = self.X[self.y==c]
            for i, attr in enumerate(x):
                probas[c].append(norm(np.mean(dat[:,i]), np.std(dat[:,i])).pdf(attr))
            probas[c].append(len(dat)/len(self.X))

        final_probas = {}
        from functools import reduce
        for c, prbs in probas.items():
            # theoretically not final probability because not divided by Pr(x)
           final_probas[reduce(lambda x,y:x * y, prbs)] = c
        return final_probas[max(final_probas)]
    
    def predict(self, X):
        return np.array([self._predict(x) for x in X])


# ============================================================================
#                            K Nearest Neighbors
# ============================================================================


class MyKNNClissifier:

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


from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    import doctest
    doctest.testmod()
