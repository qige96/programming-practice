
import numpy as np

def PCA_KLT(X, k):
    '''
    PCA for X using Karhunen-Loeve Transform. Return the top k components.
    
    Parameters
    ----------
    X: ndarray of shape (m, n)
        sample data where row represent sample and column represent feature
    k: int 
        how many components to return 

    Returns
    -------
    X_transformed
        top k component of original data
    '''
    mean_vector = np.mean(X, axis=0)
    Xp = X - mean_vector  # non-zero vectors

    def covM(_X):
        '''compute covariance matrix'''
        M = np.zeros([_X.shape[1], _X.shape[1]])
        for i in range(len(_X)):
            col_vec = _X[i].reshape([-1,1]) # the i-th column vector
            M += col_vec.T.dot(col_vec)
        return M / len(_X)

    X_cov = covM(Xp)
    E, V = np.linalg.eig(X_cov) # eigenvectors and eigenvalues
    
    projector = np.array([V.T[np.argsort(-E)[i]] for i in range(k)])

#     print('mean vector:', mean_vector)
#     print('non-zero vectors:', Xp)
#     print('Cov Mat:', X_cov)
#     print('projector:', projector)
#     print('Cov Mat of transformd data:',np.round(covM(projector.dot(Xp.T).T),5))

    return projector.dot(Xp.T).reshape(-1, k)

class MyPCA:

    def __init__(self):
        self.projector = None

    def _covM(self, _X):
        '''compute covariance matrix'''
        M = np.zeros([_X.shape[1], _X.shape[1]])
        for i in range(len(_X)):
            col_vec = _X[i].reshape([-1,1]) # the i-th column vector
            M += col_vec.T.dot(col_vec)
        return M / len(_X)

    def fit(self, X, k):
        '''
        PCA for X using Karhunen-Loeve Transform. Return the top k components.
        
        Parameters
        ----------
        X: ndarray of shape (m, n)
            sample data where row represent sample and column represent feature
        k: int 
            how many components to return 

        Returns
        -------
        X_transformed
            top k component of original data
        '''
        mean_vector = np.mean(X, axis=0)
        Xp = X - mean_vector  # non-zero vectors

        X_cov = self._covM(Xp)
        E, V = np.linalg.eig(X_cov) # eigenvectors and eigenvalues
        
        self.projector = np.array([V.T[np.argsort(-E)[i]] for i in range(k)])
        # print('mean vector:', mean_vector)
        # print('non-zero vectors:', Xp)
        # print('Cov Mat:', X_cov)
        # print('projector:', projector)
        # print('Cov Mat of transformd data:',np.round(self._covM(projector.dot(Xp.T).T),5))
        return self

    def fit_transform(self, X, k):
        '''
        PCA for X using Karhunen-Loeve Transform. Return the top k components.

        Parameters
        ----------
        X: ndarray of shape (m, n)
            sample data where row represent sample and column represent feature
        k: int 
            how many components to return 

        Returns
        -------
        X_transformed
            top k component of original data
        '''
        mean_vector = np.mean(X, axis=0)
        Xp = X - mean_vector  # non-zero vectors
        X_cov = self._covM(Xp)
        E, V = np.linalg.eig(X_cov) # eigenvectors and eigenvalues
        self.projector = np.array([V.T[np.argsort(-E)[i]] for i in range(k)])
        X_transformed = self.projector.dot(Xp.T).reshape(-1, k)
        return X_transformed
