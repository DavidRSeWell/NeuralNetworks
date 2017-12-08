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

        self.W2 = np.random.random((1,4))

        self.h = None

        self.X = None

        self.Y = None

        self.alpha =  alpha

    def load_training_date(self,size):

        x = []
        y = []

        for i in range(size):

            x1 = np.random.randint(10)

            x2 = np.random.randint(10)

            x.append([x1,x2])

            y.append(x1*x2)

        self.X = x
        self.Y = y

    def load_test_date(self, size):

        x = []
        y = []

        for i in range(size):
            x1 = np.random.randint(10)

            x2 = np.random.randint(10)

            x.append(np.array([x1, x2]))

            y.append(x1 * x2)

        self.X_test = np.array(x)

        self.Y_test = y

    def l2_loss(self,y,y_hat):

        a = np.add(y,-y_hat)**2

        return 0.5*a

    def forward(self,x):

        x = np.reshape(x,(2,1))

        layer1 = np.dot(self.W1,x)

        self.h = self.sigmoid(layer1)

        layer2 = np.dot(self.W2,self.h)

        out = self.sigmoid(layer2)

        return out

    def backward(self,y,x):

        ##### W2
        ### (y - y^)(y^ - y^2)*h

        y_hat = self.forward(x)

        y = np.reshape(y,(1,1))

        y_diff = np.dot((y - y_hat),(y_hat - y_hat**2))

        dE_dW2 = np.dot(y_diff,self.h.T)

        dWa = np.dot(np.dot(y_diff,self.W2),self.h)

        dWa1 = np.add(-1,self.h.T)

        dE_dW1= np.dot(dWa1.T,x.T)

        '''dE_dW2 = np.dot(np.add(y,-y_hat),self.h.T)

        dE_dW1_0 = np.dot(np.dot(np.add(y,-y_hat),self.W2),self.h)

        h_diff = np.add(1,-self.h).T

        dE_dW1 = np.dot(np.dot(dE_dW1_0,h_diff).T,x.T)'''

        self.W1 -= self.alpha*dE_dW1

        self.W2 -= self.alpha*dE_dW2

    def sigmoid(self,x):

        return 1.0/(1.0 + np.exp(-x))

    def sigmoid_prime(self,x):

        return self.sigmoid(x)*(1 - self.sigmoid(x))

    def total_loss(self):

        correct = 0

        loss = 0

        for i in range(len(self.X)):

            y_hat = self.forward(self.X[i])[0][0]

            loss_ = self.l2_loss(self.Y[i],y_hat)

            loss += loss_

            if y_hat == self.Y[i]:

                correct += 1

        print "Correct: " + str(correct)

        print "Loss: " + str(loss / len(self.X))