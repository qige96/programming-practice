import copy
from collections import Counter
import numpy as np 


class MyAdaBoostClassifier:

    def __init__(self, base_learner=None, max_learners=100):
        '''
        Create an AdaBoosted ensemble classifier 
        (currently only do binary classification)
        
        Parameters
        ----------
        base_learner: object
            weak learner instance that must support sample weighting when 
            training suppport fit(X, y, sample_weight), predict(X), 
            and score(X, y) methods. By default use `DecisionTreeClassifier` 
            from scikit-learn package.
        max_learners: int
            maximum number of ensembled learners
        '''
        if base_learner is None:
            from sklearn.tree import DecisionTreeClassifier
            base_learner = DecisionTreeClassifier(max_depth=1)
        # base learner must support sample weighting when training
        self.learners = [copy.deepcopy(base_learner) for k in range(max_learners)]
        self.alphas = [0] * max_learners  # weights of each learner

    def fit(self, X, y):
        '''
        Build an AdaBoosted ensemble classifier

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
        # weights of each training sample
        weights = np.ones(len(X)) * (1/len(X))
        
        for k in range(len(self.learners)):
            self.learners[k].fit(X, y, sample_weight=weights)
            y_hat = self.learners[k].predict(X)
            err_rate = 1 - self.learners[k].score(X,y) 
            if err_rate > 0.5:
                break
            alpha = 0.5 * np.log((1-err_rate)/(err_rate))
            self.alphas[k] = alpha
            weights = weights * np.exp(-alpha * y * y_hat)
            weights /= sum(weights)
 
        self.learners = self.learners[:k+1]
        self.alphas = np.array(self.alphas[:k+1]).reshape([-1,1])
        return self
    
    def predict(self, X, func_sign=None):
        '''
        Make prediction by the trained model.

        Parameters
        ----------
        X: ndarray of shape (m, n)
            data to be predicted, the same shape as trainning data
        func_sign: function
            sign function that convert weighted sums into labels

        Returns
        -------
        C: ndarray of shape (m,)
            Predicted class label per sample.
        '''
        if func_sign is None:
            def func_sign(x):
                return np.where(x<0, -1, 1)
        Y = []
        for learner in self.learners:
            Y.append(learner.predict(X))
        Y = np.array(Y)
        # weighted sum of result from each classifier
        y = np.sum(Y * self.alphas, axis=0) 
        return func_sign(y)


class MyRandomForestClassifier:

    def __init__(self, n_learners=100):
        from sklearn.tree import DecisionTreeClassifier
        base_tree = DecisionTreeClassifier(max_features='log2')
        self.learners = [copy.deepcopy(base_tree) for k in range(n_learners)]

    def fit(self, X, y):
        '''
        Build an random forest classifier

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
        for learner in self.learners:
            new_index = np.random.choice(range(len(X)), int(0.6*len(X)), replace=False)
            new_X = X[new_index]
            new_y = y[new_index]
            learner.fit(new_X, new_y)
        return self

    def predict(self, X):
        '''
        Make prediction by the trained model.

        Parameters
        ----------
        X: ndarray of shape (m, n)
            data to be predicted, the same shape as trainning data
        func_sign: function
            sign function that convert weighted sums into labels

        Returns
        -------
        C: ndarray of shape (m,)
            Predicted class label per sample.
        '''
        Y = []
        y = []
        for learner in self.learners:
            Y.append(learner.predict(X))
        Y = np.array(Y)
        
        for i in range(Y.shape[1]):
            counter = Counter(Y[:, i])
            y.append(counter.most_common(1)[0][0])

        return np.array(y)