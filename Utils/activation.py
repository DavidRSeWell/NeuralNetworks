import numpy as np


def sigmoid(array):

    pass


def tanh(array):

    pass



def softmax(x):

    denom = np.sum(np.exp(x))

    result = np.array([np.exp(x[i])/denom for i in range(len(x))])

    return result