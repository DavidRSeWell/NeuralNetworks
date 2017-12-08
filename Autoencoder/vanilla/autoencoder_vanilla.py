
import numpy as np
import sys

#####################################
# Vanilla Autoencoder implementation
#####################################

class Autoencoder_vanilla():
    '''
    Simple autoencoder with mse as the objective function
    '''

    def __init__(self,X,size=4):

        self.X = X
        self.size = size # size of the weight matrix ( size x m) m = shape[0] of X
        self.W1 = None
        self.b = None

    def init(self):

        '''
        Init the network based off the X input
        :return: null

        '''

        try:

            m_dim,n_dim = np.shape(self.X)

        except Exception,e:

            print "The data you input needs to be an array"

            sys.exit()


        W1 = np.random.random((self.size,m_dim))

        b = np.random.random((self.size,1))

        self.W1 = W1
        self.b = b


    def forward(self,x):

        try:
            np.shape(x)

        except Exception,e:
            print e

        A1 = np.dot(self.W1,x)








