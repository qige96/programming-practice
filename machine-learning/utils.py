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

def sigmoid( X): 
   return 1 / (1 + np.exp(-X))


if __name__ == '__main__':
    import doctest
    doctest.testmod()

