'''
From the paper "Why does deep and cheap learning work so well?"
By Max tegmark
https://arxiv.org/pdf/1608.08225.pdf

A simple 2 layer 4 node neural net to approximate multiplication
'''


import numpy as np


class NNmv():

    def __init__(self,alpha):

        self.W1 = np.random.random((4,2))

        self.W2 = np.random.random((4,1))

        self.h = None

        self.X = None

        self.Y = None

        self.alpha =  alpha


    def forward(self,x):

        x = np.reshape(x,(2,1))

        layer1 = np.dot(self.W1,x)

        self.h = self.sigmoid(layer1)

        layer2 = np.dot(self.W2,self.h.T)


    def l1loss(self,out,actual):

        return out - actual

    def backward(self):

        dL_W2 = self.h.copy()

        a1 = np.dot(np.add(-1,self.h),self.X)

        a2 = np.dot(self.h,a1)

        dL_W1 = np.dot(self.W2,a2)

        self.W2 -= self.alpha*dL_W2

        self.W1 -= self.alpha*dL_W1

    
    def sigmoid(self,x):

        return 1.0/(1.0 + np.exp(-x))


    def sigmoid_prime(self,x):

        return self.sigmoid(x)*(1 - self.sigmoid(x))

