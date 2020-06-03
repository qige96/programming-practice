'''
Implementation of some algorithms in PNN,
a suppliment of KCL PNN lecture notes

@ author: Ricky Zhu, Yi Li
@ email:  rickyzhu@foxmail.com
'''
from __future__ import print_function
import numpy as np
import pandas as pd

# set pandas to print more comfortable output
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


# ================================================
#             utility functions
# ================================================

def augmented_vectors(X, normalised=False):
    '''
    WARNNING: this function has beed deprecated,
                see `augmented_notation` below

    X:      2d numpy array, samples data, where rows are 
            samples and columns are features
    '''
    if normalised:
        aug_X = np.hstack([np.ones(len(X)).reshape([-1,1]), X])
        return np.array([labels[i] * aug_X[i] for i in range(len(aug_X))])
    else:
        return np.hstack([np.ones(len(X)).reshape([-1,1]), X])

def augmented_notation(X, direction='left'):
    '''
    Convert dataset into augmented notation

    X:         2d array - dataset to convert
    direction: str - concat the ones on which direction of dataset
                    must be either 'left', 'right', 'up', or 'down'
    '''
    if direction not in ['left', 'right', 'up', 'down']:
        raise KeyError("direction must be one of ['left', 'right', 'up', 'down']")
    if direction == 'left':
        return np.hstack([np.ones(X.shape[0]).reshape([-1,1]), X])
    if direction == 'right':
        return np.hstack([X, np.ones(X.shape[0]).reshape([-1,1])])
    if direction == 'up':
        return np.vstack([np.ones(X.shape[1]).reshape([1,-1]), X])
    # if direction == 'down':
    return np.vstack([X, np.ones(X.shape[1]).reshape([1,-1])])

def normalise(X, labels):
    '''normalise dataset by multiplying -1 to negative samples'''
    return np.array([labels[i] * X[i] for i in range(len(X))])

def euclidean(a, b):
    '''Euclidean distance between two vectors'''
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a-b)

def sigmoid(Wx):
    return 1 / (1+np.exp(-Wx))

def linear_function(Wx):
    return Wx

def sym_tan_sigmoid(Ws):
    return 2/(1+np.exp(-2*Ws)) - 1

def log_sigmoid(Ws):
    return 1/(1+np.exp(-Ws))

def format_logdata(log_data):
    '''formatize log data using pandas.DataFrame'''
    df = pd.DataFrame(data=log_data[1:], columns=log_data[0])
    df.set_index(log_data[0][0])
    return df


# ======================================================
#                trainning procedures
# ======================================================


def sequential_perceptron_learning_without_normlisation(
        Y, a, labels, eta=1, max_iter=10, print_log=False):
    '''
    sequential perceptron learning to train discriminant function

    Y:        2d array - training data in augmented notation
    a:        1d array - initial weights
    labels:   1d array - labels of corresponding sample in Y
    eta:      float - learning rate
    max_iter: int - max iterations to train
    print_log: bool - whether to print out log data
    '''
    log_data = [['iteration', 'a^t', 'y^t[k]','a_new', 'labels[k]', 'g(x)']]
    for i in range(max_iter):
        k = i % len(Y)
        if (a.dot(Y[k]) * labels[k]) < 0:  # misclassifed
            a_new = a + eta * labels[k] * Y[k]
        else:
            a_new  = a
        log_data.append([i+1, a, Y[k], a_new, labels[k], a.dot(Y[k])])
        a = a_new
    if print_log:
        print(format_logdata(log_data))
    return a

def sequential_perceptron_learning_with_normlisation(
        Y, a, labels, eta=1, max_iter=10, print_log=False):
    '''
    sequential perceptron learning to train discriminant function

    Y:        2d array - training data in augmented and normalised notation
    a:        1d array - initial weights
    labels:   1d array - labels of corresponding sample in Y
    eta:      float - learning rate
    max_iter: int - max iterations to train
    print_log: bool - whether to print out log data
    '''
    log_data = [['iteration', 'a^t', 'y^t[k]','a_new', 'labels[k]', 'g(x)']]
    for i in range(max_iter):
        k = i % len(Y)
        if a.dot(Y[k]) < 0:  # misclassifed
            a_new = a + eta * Y[k]
        else:
            a_new  = a
        log_data.append([i+1, a, Y[k], a_new, labels[k], a.dot(Y[k])])
        a = a_new
    if print_log:
        print(format_logdata(log_data))
    return a

def sequential_LMS_learning(Y, a, b, alpha=1, max_iter=10, print_log=False):
    '''
    Sequential Widrow-Hoff (LMS) Learning Algorithm to train
    discriminant function
    
    Y:        2d array - training data in augmented notation
    a:        1d array - initial weights
    b:        1d array - positive-valued margin vector
    eta:      float - learning rate
    max_iter: int - max iterations to train
    print_log: bool - whether to print out log data
    '''
    log_data = [['iteration', 'a^t', 'y^t_k', 'a_new', 'g(x)=ay']]
    for i in range(max_iter):
        k = i % len(Y)
        a_new = a + alpha * (b[k] - a.dot(Y[k])) * Y[k]
        log_data.append([i+1, a, Y[k], a_new, a.dot(Y[k])])
        a = a_new
    if print_log:
        print(format_logdata(log_data))
    return a

def batch_perceptron_learning_without_normalisation(
        Y, a, labels, learning_rate=0.1, max_iter=10, print_log=False):
    log_data = [['iteration', 'a^t', 'a_new' ]]
    for i in range(max_iter):
        accumulated = 0
        for k in range(len(Y)):
            if (a.dot(Y[k]) * labels[k]) < 0:  # misclassifed
                accumulated += learning_rate * labels[k] * Y[k]
        a_new = a + accumulated
        log_data.append([i+1, a, a_new])
        a = a_new
    if print_log:
        print(format_logdata(log_data))
    return a


def batch_perceptron_learning_with_normalisation(
        Y, a, labels, learning_rate=0.1, max_iter=10, print_log=False):
    log_data = [['iteration', 'a^t', 'a_new' ]]
    for i in range(max_iter):
        accumulated = 0
        for k in range(len(Y)):
            if a.dot(Y[k]) < 0:  # misclassifed
                accumulated += learning_rate * Y[k]
        a_new = a + accumulated
        log_data.append([i+1, a, a_new])
        a = a_new
    if print_log:
        print(format_logdata(log_data))
    return a

def batch_LMS_learning(Y, a, b, learning_rate=0.1, max_iter=10, print_log=False):
    '''
    Widrow-Hoff (LMS) method
    '''
    log_data = [['iteration', 'a^t', 'a_new']]
    for i in range(max_iter):
        a_new = a - learning_rate * Y.T.dot(Y.dot(a) - b)
        log_data.append([i+1, a, a_new])
        a = a_new
    if print_log:
        print(format_logdata(log_data))
    return a

# X = np.array([
#     [0,0],
#     [1,0],
#     [2,1],
#     [0,1],
#     [1,2]
# ])
# labels = np.array([1,1,1,-1,-1])
# a = np.array([-1.5, 5, -1])
# Y1= augmented_notation(X)
# batch_perceptron_learning(Y1, a, labels, 1, True)

# b = np.array([2,2,2,2,2])
# Y2 = normalise(augmented_notation(X), labels)
# batch_LMS_learning(Y2, a, b, 0.1, 1000)
# sequential_LMS_learning(Y2, a, b, 0.2 ,10)

# X = np.array([
#     [0,2],
#     [1,2],
#     [3,1],
#     [-3,1],
#     [-2,-1],
#     [-3, -2]
# ])
# labels = np.array([1,1,1,-1,-1,-1])
# a = np.array([1,0,0])
# Y1= augmented_notation(X)
# sequential_perceptron_learning_without_normlisation(
#     Y1, a, labels, eta=1, max_iter=13, print_log=True)

def sequential_delta_learning(X, w, labels, eta=0.1, max_iter=10, print_log=False):
    '''
    sequential delta learning for Linear Threshold Unit

    X:        2d array - training data in augmented notation
    w:        1d array - initial weights
    labels:   1d array - labels of corresponding sample in Y
    eta:      float - learning rate
    max_iter: int - max iterations to train
    print_log: bool - whether to print out log data
    '''
    log_data = [['iteration', 'label[k]', 'X[k]', 'w', 'H(wx)',
        'eta*(label[k]-H(wx))', 'w_new']]
    for i in range(max_iter):
        def H(wx):
            if wx > 0: return 1
            else: return 0
        k = i % len(X)
        w_new = w + eta * (labels[k] - H(w.dot(X[k]))) * X[k]
        log_data.append([i+1, labels[k], X[k], w, H(w.dot(X[k])),
            eta*(labels[k]-H(w.dot(X[k]))), w_new])
        w = w_new
    if print_log:
        print(format_logdata(log_data))
    return w

def sequential_hebbian_learning(x, W, alpha=0.1, max_iter=10, print_log=False):
    '''sequential Hebbian learning for negative feedback network'''
    log_data = [['iteration', 'e', 'We', 'y', 'Wy']]
    e = x
    y = np.zeros(W.shape[0])
    for i in range(max_iter):
        y = y + alpha*W.dot(e)
        log_data.append([i+1, e, W.dot(e), y, W.T.dot(y)])
        e = x - W.T.dot(y).T
    if print_log:
        print(format_logdata(log_data))
    return y

# X = np.array([
#     [0,2],
#     [1,2],
#     [2,1],
#     [-3,1],
#     [-2,-1],
#     [-3,-2]
# ])
# labels = np.array([1,1,1,0,0,0])
# w = np.array([1, 0, 0])
# aug_X = augmented_notation(X)
# sequential_delta_learning(aug_X, w, labels, 1, 13, True)

# W = np.array([[1,1,0], [1,1,1]])
# x = np.array([1,1,0])
# sequential_hebbian_learning(x, W, 0.25, 5, True)

# =============================================
#            classifier models
# =============================================

def linear_discrminant_function(X, w, w0):
    return w.dot(X.T) + w0

# X = np.array([
#     [1,1],
#     [2,2],
#     [3,3]
# ])
# w = np.array([2, 1])
# print(linear_discrminant_function(X, w, -5))

def _top_k_nearest_neightbors(centroid, surroundinds, k, func_dist=None, print_log=False):
    '''return indices of the top k nearest neighbors of centroid from surroundings'''
    if func_dist==None:
        func_dist = np.linalg.norm
    distances = []
    for vec in surroundinds:
        distances.append(func_dist(centroid - vec))
    distances = np.array(distances)
    top_k = np.argsort(-distances)[:k]
    if print_log:
        print('neightbors and distances:', [(v, d) for v, d in zip(surroundinds, distances)])
    return top_k

def knn(X_train, y_train, x, k, func_dist=None, print_log=False):
    '''
    K nearest neightbors.
    
    X_train:   2d array - training data
    y_train:   1d array - labels of training samples
    x:         1d array - the sample whose label needs to determine
    k:         int - how many nearest neightbors to count
    func_dist: function - function to compute distance between samples, by default Eulidean 
    print_log: bool - whether to print out intermediate results
    '''
    if func_dist==None:
        func_dist = np.linalg.norm
    top_k_indices = _top_k_nearest_neightbors(x, X_train, k, func_dist, print_log)
    top_k = X_train[top_k_indices]
    top_k_labels = y_train[top_k_indices]
    if print_log:
        print('top k neightbors:', top_k)
        print('class of top k neightbors:', top_k_labels)
    from collections import Counter
    return Counter(top_k_labels).most_common(1)[0][0]

# surroundinds = np.array(
#     [0.15, 0.35],
#     [0.15, 0.28],
#     [0.12, 0.2],
#     [0.1, 0.32],
#     [0.06, 0.25]
# ])
# labels = np.array([1,2,2,3,3])
# print(_top_k_nearest_neightbors([0.15, 0.25], surroundinds, 3))
# print(knn(surroundinds, labels, [0.15, 0.25], 1, None, True))V

def MLP(x, Ws, func_a, print_log=False):
    '''
    apply multilayer perceptron to x using determined weights
    assume that in input layer the activation function is linear function
    
    x:         1d array - input sample
    Ws:        list of matrix - array of weights for each layer
    func_a:    list of function - array of activation function for each layer,
                excluding input layer
    print_log: bool - whether to print out intermediate results
    '''
    for i in range(len(Ws)):
        aug_x = np.concatenate([x, [1]])
        yi = Ws[i].dot(aug_x)
        ai = func_a[i](yi)
        if print_log:
            print('output of layer', i+1, ':', ai)
        x = ai
    return x

# W1 = np.array([
#     [-0.7057, 1.9061, 2.6605, -1.1359, 4.8432],
#     [0.4900, 1.9324, -0.4269, -5.1570, 0.3973],
#     [0.9438, -5.4160, -0.3431, -0.2931, 2.1761]
# ])
# W2 = np.array([
#     [-1.1444, 0.3115, -9.9812, 2.5230],
#     [0.0106, 11.5477, 2.6479, 2.6463]
# ])
# x = np.concatenate([np.array([1, 0, 1, 0]), [1]])
# def linear_function(Wx):
#     return Wx
# def sym_tan_sigmoid(Ws):
#     return 2/(1+np.exp(-2*Ws)) - 1
# def log_sigmoid(Ws):
#     return 1/(1+np.exp(-Ws))
# func_a = [ sym_tan_sigmoid, log_sigmoid]
# print(MLP(np.array([1, 0, 1, 0]), [W1, W2], func_a))
# print(MLP(np.array([0, 1, 0, 1]), [W1, W2], func_a))
# print(MLP(np.array([1, 1, 0, 0]), [W1, W2], func_a, True))


def RBF_net(X, C, W, func_h, func_f, print_log=False):
    '''
    Radial Basis Function Neural Network

    X:         2d array - sample data
    C:         2d array - predefiend centroids of hidden layer
    W:         2d array - predefiend weights of output layer
    func_h:    function - activation function of hidden layer
    func_f:    function - activation function of output layer
    print_log: bool - whether to print out intermediate results
    '''
    d = np.array([np.linalg.norm(X - c, axis=1) for c in C])
    Y = func_h(d)
    aug_Y = augmented_notation(Y, 'down')
    Z = func_f(W.dot(aug_Y))
    if print_log:
        print('d:', d)
        print('Y:', Y)
    return Z

# X = np.array([
#     [0,0],
#     [0,1],
#     [1,0],
#     [1,1]
# ])
# C = X[[0,3]]
# W = np.array([[-2.5031, -2.5031,  2.8418]])
# def f_h(d):
#     return np.exp(-(d**2)/(2*(1/np.sqrt(2))**2))
# print(RBF_net(X, C, W, f_h, linear_function).round(2))


def _gradient_h_o(t, z, fp_net2, y):
    return (t-z) * fp_net2 * y
def update_hidden_output_weights(w, eta, t, z, fp_net2, y):
    return w + eta * _gradient_h_o(t, z, fp_net2, y)

def _gradient_i_h(t, z, fp_net1, fp_net2, w2, x):
#     print((t-z) * fp_net2 * w2, fp_net1, x)
    return (t-z) * fp_net2 * w2 * fp_net1 * x
def update_input_hidden_weights(w1, eta, t, z, fp_net1, fp_net2, w2, x):
#     print( _gradient_i_h(t, z, fp_net1, fp_net2, w2, x))
    return w1.T + eta * _gradient_i_h(t, z, fp_net1, fp_net2, w2, x)

# def f(Ws):
#     return 2/(1+np.exp(-2*Ws)) - 1
# def fp(net):
#     return (4*np.exp(-2*net)) / (1 + np.exp(-2*net))**2
# def net(W, x):
#     return W.dot(x)
# W1 = np.array([
#     [0.2, 0.5, 0],
#     [0, 0.3, -0.7],
# ])
# W2 = np.array([-0.4, 0.8, 1.6])
# x = np.array([0.1, 0.9])
# t = 0.5

# net1 = net(W1, np.concatenate([[1], x]))
# y = f(net1)
# print('net1:', net1)
# print('y:', y)
# net2 = net(W2.T, np.concatenate([[1], y]))
# z = f(net2)
# print('net2:', net2)
# print('z:', z)

# print(update_input_hidden_weights(W1[1,1], 0.25, t, z, fp(net1)[1], fp(net2), W2[2], x[0]))


def compute_output_weights_RBF(aug_Y,t):
    '''
    determine weights of output layer for RBF net
    
    aug_Y: 2d array - augmented Y (concat ones on the right),
                    outputs of RBF hidden layer
    t:     1d array - labels of samples
    '''
    adjusted_Y = aug_Y.transpose().dot(aug_Y)
    return np.linalg.inv(adjusted_Y).dot(aug_Y.transpose()).dot(t).round(4)

# Y = np.array([
#     [1.0000,    0.1353,    1.0000],
#     [0.3679,    0.3679,    1.0000],
#     [0.3679,    0.3679,    1.0000],
#     [0.1353,    1.0000,    1.0000]
# ])
# t = np.array([0,1,1,0])
# print(compute_output_weights_RBF(Y,t))


def compute_GAN_cost(real, fake, func_D):
    Ex = np.sum([np.log(func_D(x))/len(real) for x in real])
    Ez = np.sum([np.log(1-func_D(x))/len(fake) for x in fake])
    print(Ex, Ez)
    return Ex + Ez

# real = np.array([[1,2], [3,4]])
# fake = np.array([[5,6], [7,8]])
# def D(x):
#     return 1 / (1 + np.exp(-(0.1*x[0]-0.2*x[1]-2)))

# compute_GAN_cost(real, fake, D)

# =========================================================
#                   feature extraction
# =========================================================

def PCA_KLT(X, k, print_log=False):
    '''
    PCA for X using Karhunen-Loeve Transform. Return the top k components.
    
    X: 2darray - r by c matrix where row represent sample and column represent feature
    k: int - how many components to return 
    print_log: bool - if print out all intermediate results
    '''
    mean_vector = np.mean(X, axis=0)
    Xp = X - mean_vector
    def covM(_X):
        M = np.zeros([_X.shape[1], _X.shape[1]])
        for i in range(len(_X)):
            M += _X[i].reshape([-1,1]).dot(_X[i].reshape([1,-1]))
        return M / len(_X)
    X_cov = covM(Xp)
    E, V = np.linalg.eig(X_cov)
    print([V.T[np.argsort(-E)[i]] for i in range(k)])
    projector = np.array([V.T[np.argsort(-E)[i]] for i in range(k)])
    if print_log:
        print('mean vector:', mean_vector)
        print('non-zero vectors:', Xp)
        print('Cov Mat:', X_cov)
        print('projector:', projector)
        print('Cov Mat of transformd data:',np.round(covM(projector.dot(Xp.T).T),5))
    return projector.dot(Xp.T)

# x1 = np.array([1,2,1])
# x2 = np.array([2,3,1])
# x3 = np.array([3,5,1])
# x4 = np.array([2,2,1])
# X = np.array([x1,x2,x3,x4])
# PCA_KLT(X, k=2, print_log=True)

def batch_oja_learning(X, w, eta=0.01, epoch=2, print_log=False):
    '''
    Oja batch learning for neural network PCA

    X:        2d array - r by c matrix where row represent 
                  sample and column represent feature
    w:        1d array - initial weights
    eta:      float - learning rate
    epoch:    int - max epoch to train
    print_log: bool - whether to print out intermediate results
    '''
    mean_vector = np.mean(X, axis=0)
    Xp = X - mean_vector
    logdata = [['epoch', 'Xp[t]', 'y=wx', 'X[t]-yw', 'eta* y * (X[t]-yw)']]
    for i in range(epoch):
        delta_w = 0
        
        for t in range(len(Xp)):
            intermediates = [i+1]
            y = w.dot(Xp[t])
            intermediates.append(Xp[t])
            intermediates.append(y)
            intermediates.append(np.round(Xp[t].T - y*w, 4))
            intermediates.append(np.round(eta*y*(Xp[t].T - y*w), 4))
            delta_w += eta*y*(Xp[t].T - y*w)
            logdata.append(intermediates)
        w = w + delta_w
    if print_log:
        print(format_logdata(logdata))
    return w

# X = np.array([
#     [0,1],
#     [3,5],
#     [5,4],
#     [5,6],
#     [8,7],
#     [9,7]
# ])
# w=np.array([-1, 0])
# batch_oja_learning(X, w, 0.01, 2, True)

def LDA_J(w, X, labels, print_log=False):
    '''
    compute the cost of Fisher's LDA, only for binary classification

    w:         1d array - weights that project sample to a line
    X:         2d array - sample data, row as sample and col as feature
    labels:    1d array - labels of corresponding sample
    print_log: bool - whether to print out intermediate results
    '''
    classes = list(set(labels))
    M = np.array([np.mean(X[labels==y], axis=0) for y in classes])
    sb = w.dot(M[0] - M[1]) ** 2
    sw = np.sum([w.dot(x-M[0])**2 for x in X[labels==classes[0]]]) \
        + np.sum([w.dot(x-M[1])**2 for x in X[labels==classes[1]]])
    if print_log:
        print('sb:', sb, ', sw:', sw)
    return sb / sw

# X = np.array([
#     [1,2],
#     [2,1],
#     [3,3],
#     [6,5],
#     [7,8]
# ])
# labels = np.array([1,1,1,2,2])
# w1 = np.array([-1, 5])
# w2 = np.array([2, -3])

# print(LDA_J(w1, X, labels, True))
# print(LDA_J(w2, X, labels, True))

def extreme_learning_machine(X, V, w, func_g=None, print_log=False):
    '''
    random projection using extreme learning machine

    X:        2d array - sample data, COLOUM as sample, and ROW as feature
    V:        2d array - weights that project x to y
    w:        1d array - weights that map y to t(targets of sample)
    func_g:   function - that g(Vx) that map x to y
    print_log: bool - whether to print out intermediate results
    ''' 
    if func_g == None:
        def func_g(X, V):
            return np.where(V.dot(X)<0, 0, 1)
    Y = func_g(X, V)
    if print_log:
        print('Y (output of hidden layer): ')
        print(Y)
    return w.dot(np.vstack([np.ones(Y.shape[1]),Y]))

# V = np.array([
#     [-0.62, 0.44, -0.91],
#     [-0.81, -0.09, 0.02],
#     [0.74, -0.91, -0.60],
#     [-0.82, -0.92, 0.71],
#     [-0.26, 0.68, 0.15],
#     [0.80, -0.94, -0.83],
# ])
# X = np.array([
#     [1,1,1,1],
#     [0,0,1,1],
#     [0,1,0,1]
# ])
# w = np.array([0,0,0,-1,0,0,2])
# print(extreme_learning_machine(X,V,w,print_log=True))

def spaCodRecErr(x, V, y):
    '''
    reconstruction error for dictionary-based sparse coding
    
    x: a dense vector
    V: the dictionary
    y: the transfered sparse coding of x
    '''
    return np.linalg.norm(x - V.dot(y))


# ========================================================
#               support vector machine
# ========================================================

def compute_svm_weights(X_supp, Y_supp, print_log=False):
    '''
    compute weights of a linear SVM with identified support vectors

    X_supp: 2d array - each row represents a support vector
    Y_supp: 1d array - label of each support vector, either 1 or -1
    '''
    A = X_supp.dot(X_supp.T)*Y_supp
    A = np.vstack([A, Y_supp])
    A = np.hstack([A, np.array([1]*len(X_supp)+[0]).reshape([-1,1])])
    b = np.concatenate([Y_supp, [0]])
    res = np.linalg.inv(A).dot(b)
    lambdas, w0 = res[:-1], res[-1]
    w = np.sum([lambdas[i]*X_supp[i]*Y_supp[i] for i in range(len(X_supp))], axis=0)
    if print_log:
        print('euqation atrix:', A)
        print('equation vector:', b)
        print('lambdas:',lambdas, ', w0:', w0)
    return (w, w0)

# X_supp = np.array([
#      [3,1],
#      [3,-1],
#      [1,0]
#      ]
# )
# Y_supp = np.array([1,1,-1])
# print(compute_svm_weights(X_supp, Y_supp, print_log=True))


# =============================================================
#                      tree and forest
# =============================================================

'''count number of wrong classification'''
def calculate_error(classfier,a_d,min_class_d):
    lens=len(a_d)
    print('current best classifier(h_bar): \ncoiffician:' ),
    print (a_d)
    print('classifier:')#multiple correspoding variables
    print(min_class_d)
    kmax = classifier.shape[1]
    x=np.zeros(kmax)
    for i in range(lens):
        x=x+(a_d[i]*classifier[min_class_d[i]])
    x=np.sign(x)
    return ((classifier[0]!=x).sum())


#adaboost
def adaboost(classifier):
    kmax=classifier.shape[0]-1
    n=classifier.shape[1]
    w=np.full(n,1/n)
    w_tmp=np.zeros(n)
    a_d=np.array([0.])
    min_class_d=np.array([0])
    for k in range(kmax):
        print('interation: %d' %(k+1))
        emin=1.
        min_classifier=0
        #a_d=np.array([0])
        #min_class_d=np.array([0])
        for i in range(kmax):
            e=0.
            wrong_loc=np.where(classifier[0]!=classifier[i+1])
            for j in range(len(wrong_loc[0])):
                e=round(e+w[wrong_loc[0][j]],4)
            #print (e), #weighted training error
            if e<emin:
                emin=e
                min_classifier=i+1
        #print('emin=%.4f' %emin)
        #need change to weight train error
        if emin>0.5:
            kmax=k-1
            break
        a=round((1/2)*(np.log((1-emin)/emin)),4)
        if k==0:
            a_d[0]=a
            min_class_d[0]=min_classifier
        else:
            a_d=np.append(a_d,[a])
            min_class_d=np.append(min_class_d,[min_classifier])
        print('a=%.4f' %a)
        error_num=calculate_error(classifier,a_d,min_class_d)
        #print ('eu%d' %error_num)
        if error_num==0:
            break
        for i in range(n):
            w_tmp[i]=round(w[i]*np.exp(-a*classifier[0][i]*classifier[min_classifier][i]),4)
        #print('wki*e^...')
        #print(w_tmp)
        z=w_tmp.sum()
        #print ('z=%.4f' %z)
        w=np.around(w_tmp/z,4)
        print('w(k+1):')
        print(w)
    print ('END')
    
classifier_number=8
'''
classifier[0]->real classes [1]->classsifier1 [2]->classsifier2 ...
'''
'''
classifier=np.array([[1,1,1,1,1,-1,-1,-1,-1,-1],[1,1,-1,-1,-1,-1,-1,-1,-1,-1],
                     [1,1,1,1,1,1,1,1,-1,-1],[-1,-1,1,1,1,-1,-1,-1,1,-1]])

'''    
classifier=np.array([[1,1,-1,-1],[1,-1,1,1],[-1,1,-1,-1],[1,-1,-1,-1],[-1,1,1,1],
                     [1,1,1,-1],[-1,-1,-1,1],[-1,-1,1,-1],[1,1,-1,1]])
adaboost(classifier)


# ==============================================================
#                       clustering
# ==============================================================


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

def my_kmeans_with_centres(X, centers, max_iter=3, print_log=False):
    '''
    My k-means algorithm
    
    X:         2d array - sample data
    centres:   2d array - predefined centroids
    max_iter:  int - max iteration
    print_log: bool - whether to print out intermediate results
    '''
    prev_labels = np.zeros(len(X))
    for i in range(max_iter):
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
    return labels

# S = np.array([
#     [-1,3],
#     [1,4],
#     [0,5],
#     [4,-1],
#     [3,0],
#     [5,1]
# ])
# centres = np.array([[-1,3, 1], [5,1, 1]])
# my_kmeans_with_centres(augmented_notation(S), centres, print_log=True)
    
def distance(x,y):
    return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)


def distance_3(x,y):
    return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2+(x[2]-y[2])**2)
'''
x=np.array([-1.,3.])
y=np.array([0.5,2.5])
print(distance(x,y))
'''


def k_means(c,m,x):
    '''
    c:number of cluster centers
    m: initialise cluster center position
    x: sample datas the last colomns is the label, others is features
    '''
#    assigned_c=np.array([0,0,0,0,0,0])
    for i in range(c):
        for j in range(x.shape[0]):
            if distance(x[j],m[0])<distance(x[j],m[1]):
                x[j][2]=1
            else:
                x[j][2]=2
        m[0]=np.mean(x[x[:,2]==1,:],axis=0)
        m[1]=np.mean(x[x[:,2]==2,:],axis=0)
        print('interation: %d' %(i+1))
        print ('sample (the last colomns is the label):')
        print(x)
        print('cluster center(no meaning for the last variable):')
        print(m)

'''
#initialise
m=np.array([[-1,3,0],[5,1,0]],dtype=float)# no meaning for the last variable
x=np.array([[-1,3,0],[1,4,0],[0,5,0],[4,-1,0],[3,0,0],[5,1,0]],dtype=float)
c=2
k_means(c,m,x)
'''

#tutorial10 q5
def fuzzy_kmeans(c,x,u):
    m=np.array([[0.,0.],[0.,0.]])
    b=u.shape[1]
    #nromalised u
    u_sum=u.sum(axis=1)
    u_norm=u.copy()
    u_norm[:,0]= u[:,0]/u_sum
    u_norm[:,1]= u[:,1]/u_sum
    for i in range(c):
        #upgrade m
        u_norm=np.power(u_norm,b)
        m[0][0]=round(np.average(x[:,0],weights=u_norm[:,0]),4)
        m[0][1]=round(np.average(x[:,1],weights=u_norm[:,0]),4)
        m[1][0]=round(np.average(x[:,0],weights=u_norm[:,1]),4)
        m[1][1]=round(np.average(x[:,1],weights=u_norm[:,1]),4)
        print('interation: %d' %(i+1))
        print(m)
        #upgrade u
        tmp=2/(b-1)
        for j in range(u.shape[0]):
            u_norm[j][0]=round((1/distance(x[j],m[0]))**tmp/((1/distance(x[j],m[0]))**tmp+(1/distance(x[j],m[1]))**tmp),4)
            u_norm[j][1]=round((1/distance(x[j],m[1]))**tmp/((1/distance(x[j],m[0]))**tmp+(1/distance(x[j],m[1]))**tmp),4)
        print(u_norm)


u=np.array([[1,0],[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5],[0,1]])
x=np.array([[-1,3],[1,4],[0,5],[4,-1],[3,0],[5,1]])
#fuzzy_kmeans(3,x,u)


#interative optimisation:
def pj(n,x,m):
    #compare these two/n distance, move it if 1st smaller that 2nd
    print(n/(n+1)*distance(x,m))# move in 
    print(n/(n-1)*distance(x,m))# move out
#n:number of points in this cluster
#x:points vector
#m:cluster center    
#pg(n,x,m)
    
def distance_between_cluster(x):
    len=x.shape[0]
    dmin=100000.
    dmin_loc=np.array([0,0])
    distances=np.zeros([len,len])
    for i in range(len-1):
        for j in range (len-i):
            d=distance(x[i],x[j+i])
            distances[i][j+i]=d
            if ((d<dmin)&(i!=j)):
                dmin=d
                dmin_loc[0]=i
                dmin_loc[1]=j  
    distances=np.transpose(distances)
    #print(np.around(distances,decimals=4))
   #  print (dmin)
    return (dmin_loc,distances)

def agglomerative_hierarchical_cluster(c,x,cluster):
    a=x.shape[0]-c
    dmin,distances=distance_between_cluster(x) 
    distances[distances==0]=float('inf')
    for i in range (a):
        print('interation: %d' %(i+1))
        print('minimum distance:'),
        print(np.min(distances))
        dmin_loc=np.where(distances==np.min(distances))
        print ('minmimum location:')
        print(dmin_loc[0][0],dmin_loc[1][0])
        dmin[0]=dmin_loc[0][0]
        dmin[1]=dmin_loc[1][0]
        cluster[dmin[0]]=cluster[dmin[0]] +'+' +  cluster[dmin[1]]
        #cluster=np.delete(cluster,dmin[1],axis=0)
        print(cluster)
        distances[dmin[0]][dmin[1]]=float('inf')
        print(distances)
    print('END')

'''  
x=np.array([[-1,3,1],[1,2,2],[0,1,3],[4,0,4],[5,4,5],[3,2,6]],dtype=float)
cluster=['1','2','3','4','5','6']
c=3
#some bug in the appearence of clusters
agglomerative_hierarchical_cluster(c,x,cluster)
'''


def find_cluster(x,m):
    
    lens=m.shape[0]
    distances=np.zeros(lens)
    cluster=np.zeros(x.shape[0],dtype=int)
    number=x.shape[0]
    for i in range (number):
        for j in range(lens):
            distances[j]=distance(x[i],m[j])
        cluster[i]=(np.where(distances==np.min(distances)))[0]+1
        
    print(cluster)

# eta is learning rate
def competitive(eta,x,m,order):
    distances=np.zeros(m.shape[0])
    c=len(order)
    for i in range(c):
        current=order[i]-1
        distances[0]=distance(x[current],m[0])
        distances[1]=distance(x[current],m[1])
        distances[2]=distance(x[current],m[2])
        dmin=np.where(distances==np.min(distances))
        #print(dmin[0][0]+1)
        m[dmin[0][0]]=m[dmin[0][0]]+eta*(np.subtract(x[current],m[dmin[0][0]]))
        #print(m[dmin[0][0]])
    print(m)
    
    return(m)
    
'''
#initialise
x=np.array([[-1,3],[1,4],[0,5],[4,-1],[3,0],[5,1]],dtype=float)
m=np.zeros([3,2])
m[0]=x[0]/2
m[1]=x[2]/2
m[2]=x[4]/2
order=np.array([3,1,1,5,6])
eta=0.1
m_new=competitive(eta,x,m,order)
find_cluster(x,m_new)
x_new=np.array([[0,-2]])
find_cluster(x_new,m_new)
'''


def competitive_norm(eta,c,x,m,order):
    print ('just framework, no confirmed')
    add= np.array([1,1,1,1,1,1])
    x = np.insert(x, 0, values=add, axis=1)
    print(x)
    x_normed = x.copy()
    for i in range(len(x_normed)):
        x_normed[i]=x_normed[i]/distance_3(x_normed[i],[0,0,0])
    for i in range(5):
        '''no data to prove these codes'''
        #j=argmax mjT*x
        #arg=np.multiply(np.transpose(m),x)
        #j=np.where(arg==np.max(arg))
        #m[j]=m[j]+eta*x
        #m[j]=m[j]/distance_3(m[j],[0,0,0])
    print(x_normed)
    # distances=np.array([0.,0.,0.])
'''
    for i in range(5):
        current=order[i]-1
        distances[0]=distance(x[current],m[0])
        distances[1]=distance(x[current],m[1])
        distances[2]=distance(x[current],m[2])
        dmin=np.where(distances==np.min(distances))
        print(dmin[0][0]+1)
        m[dmin[0][0]]=m[dmin[0][0]]+eta*(np.subtract(x[current],m[dmin[0][0]]))
        print(m[dmin[0][0]])
    '''
        
        
#competitive_norm(eta,c,x,m,order)
    
    

