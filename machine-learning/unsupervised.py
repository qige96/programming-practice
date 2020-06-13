'''
Simple implementation of some supervised learning algorithms.
'''
import numpy as np

# ============================================================================
#                                   K-means
# ============================================================================


def my_kmeans(X, K, print_log=True):
    '''
    My k-means algorithm
    
    :param X: n by 2 matrix
    :param K: number of clusters
    '''
    ran_index = np.random.choice(np.arange(0,len(X)),K)
    centres = [X[i] for i in ran_index]
    prev_labels = np.zeros(len(X))
    i = 0
    while True:
        if print_log:
            print('iteration',i, ', centers: ')
            print(centres)
            print('---------------------------')
        labels = []
        for x in X:
            labels.append(np.argmin([euclidean(x, c) for c in centres]))
        if np.array_equal(prev_labels, labels):
            break
        prev_labels = labels
        for i in range(len(centres)):
            centres[i] = np.mean(X[np.array(labels)==i], axis=0)
        i += 0
    return labels

class MyKMeans:
    def __init__(self, k=3):
        '''
        K-means clustering model

        :k: int, number of clusters
        '''
        self.k = k
        self.X = None
        self.labels = None
        self.cluster_centers = None
    
    def fit(self, X):
        '''
        Train the K-means clustering model

        Parameters
        ----------
        X: ndarray of shape (m, n)
            sample data where row represent sample and column represent feature

        Returns
        -------
        self
            trained model
        '''
        ran_index = np.random.choice(np.arange(0,len(X)), self.k)
        centres = [X[i] for i in ran_index]
        prev_labels = np.zeros(len(X))
        i = 0
        while True:
            labels = []
            for x in X:
                labels.append(np.argmin([np.linalg.norm(x - c) for c in centres]))
            if np.array_equal(prev_labels, labels):
                break
            prev_labels = labels
            for i in range(len(centres)):
                centres[i] = np.mean(X[np.array(labels)==i], axis=0)
            i += 0
        self.labels = np.array(labels)
        self.cluster_centers = np.array(centres)
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
        labels = []
        for x in X:
            labels.append(np.argmin([
                np.linalg.norm(x - c) for c in self.cluster_centers
            ]))
        return np.array(labels)

# ============================================================================
#                                Incremental
# ============================================================================



# ============================================================================
#                               Hierarchical
# ============================================================================



# ============================================================================
#                           Gaussian Mixture Model
# ============================================================================

