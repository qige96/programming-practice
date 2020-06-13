'''
Some common utility functions for ML tasks.
'''
import numpy as np

def binarize(y, threashold=0.5):
    '''
    Transform numeric data into binarize 0 and 1

    Parameters
    ----------
    y: array-like
        numeric data array
    threashold: float
        data that are smaller than threashold would be converted to 0,
        otherwise 1
    
    Returns
    -------
    1d array

    Examples
    --------
    >>> y = np.array([0.27, 0.07, 0.56, 0.35, 0.32, 0.65])
    >>> binarize(y)
    array([0, 0, 1, 0, 0, 1])
    >>> y2 = [1, 2, 3, 4, 5, 6]
    >>> binarize(y2, threashold=3.5)
    array([0, 0, 0, 1, 1, 1])
    '''
    return np.where(np.array(y) < threashold, 0, 1)

def sigmoid(x): 
    '''
    Batch version Sigmoid function

    Parameters
    ----------
    X: array-like
        numeric data array
    
    Returns
    -------
    1d array

    Examples
    --------
    >>> x = np.array([-0.27, 0.07, 0.56, -0.35, 0.32, -0.65])
    >>> sigmoid(x)
    array([0.4329071 , 0.51749286, 0.63645254, 0.41338242, 0.57932425,
       0.34298954])
    '''
    return 1 / (1 + np.exp(-x))

def relu(X):
    '''
    Batch version ReLU function

    Parameters
    ----------
    X: array-like
        numeric data array
    
    Returns
    -------
    1d array

    Examples
    --------
    >>> x = np.array([-0.27, 0.07, 0.56, -0.35, 0.32, -0.65])
    >>> relu(x)
    array([0.  , 0.07, 0.56, 0.  , 0.32, 0.  ])
    '''
    return np.where(x<0, 0, x)
    

if __name__ == '__main__':
    import doctest
    doctest.testmod()

