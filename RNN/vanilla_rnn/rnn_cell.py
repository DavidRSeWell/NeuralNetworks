# -*- coding: utf-8 -*-


import numpy as np
import sys

sys.path.append('/Users/befeltingu/NeuralNetworks')

import utils


class RNNCell():

    def __init__(self,Wh,Ux,Vy,h_dim,words=None,window=1):

        self.Wh = Wh

        self.Ux = Ux

        self.Vy = Vy

        self.h = np.zeros((h_dim,1))

        self.h_prev = np.zeros((h_dim,1))

        self.h_next = np.zeros((h_dim,1))

        self.window = window

        self.words = words

    def forward_prop(self,X):
        '''
        Takes in a series of X inputs
        and returns a list of the information
        stored for eacch time step. In this case
        it is just the hidden units at that time step
        :param X:
        :return:
        '''

        self.h_prev = np.zeros(self.h.shape) # for first time step the hidden layer is zero

        layers = []

        # loop over each of the inputs performing a forward pass on each
        # and tracking the hidden layer,output for each which will be used in the
        # back prop step
        for i in range(len(X)):

            o_t = self.forward_pass(X[i])

            curr_h = self.h.copy()

            layers.append([X[i],curr_h,o_t])

        return layers

    def forward_pass(self,X):

        # a(t) = b + Wh(t−1)+ Ux(t)j - activation layer

        X = np.reshape(X,(len(X),1))

        a1 = np.dot(self.Wh,self.h_prev)

        a2 = np.dot(self.Ux,X)

        a = np.add(a1,a2)

        # h(t) = tanh(a(t)

        self.h = np.tanh(a)

        # o(t) = c + Vh(t)

        o_t = np.dot(self.Vy,self.h)

        # y(t)= softmax(o(t))

        y_hat = utils.softmax(o_t)

        return o_t

    def backprop_tt(self,X,Y):

        '''
        In order to update our weights we need the derivative
        of our loss function w.r.t to each set of weights

        dL/dV = dL_do * h_t

        dL/dW = (1 - h_t^2) * dL_dh_t * h_t_1

        dL/dU = (1 - h_t^2) * dL_dh_t * x

        The dW and dU both depend on the derivative of the hidden layer
        which is a recursive definition

        dL_dh = W * dL / (dh (t + 1) * (1 - h(t + 1) ^2) + V * dL_do

        the basic strategy is to compute all the derivates at the final timestep
        and then work backward to the original timestep

        :param x:
        :param hidden_t_1:
        :return:
        '''

        dV = np.zeros(self.Vy.shape)

        dW = np.zeros(self.Wh.shape)

        dU = np.zeros(self.Ux.shape)

        layers = self.forward_prop(X) # perform a full forward pass of the network

        h_prev = np.zeros(self.h_prev.shape)

        for t in range(len(layers)):

            h_current = layers[t][1]

            y_actual = np.reshape(Y[t],(len(X[0]),1))

            o_t_current = layers[t][2] # get output from the current layer

            y_hat_current = utils.softmax(o_t_current)

            input = np.reshape(layers[t][0],(len(X[0]),1))

            dL_do = np.add(-y_actual,y_hat_current)

            ##### dL_dh = derevative of Loss w.r.t hidden layer

            #### at the last time step the derivative of the hiddent layer
            #### only depends on the output

            dl_dh_t = np.dot(self.Vy.T,dL_do)

            dL_dV = np.dot(dL_do,h_current.T)

            dL_dW = self.delta_LW(h_current,dl_dh_t,h_prev)

            dL_dU = self.delta_LW(h_current,dl_dh_t,input)

            dV += dL_dV

            dW += dL_dW

            dU += dL_dU

            prev_dh = dl_dh_t

            for i in range(t - 1,max(-1,t - self.window -1) , -1): # back prop throught the remaining layers

                input = np.reshape(layers[i][0],input.shape)

                h_current = layers[i][1]

                h_prev = layers[i - 1][1]

                h_next = layers[i + 1][1]

                o_t = layers[i][2]

                # dL/do - derivative of Loss w.r.t output

                y_hat_t = utils.softmax(o_t)

                y_actual = np.reshape(Y[i],y_actual.shape)

                dL_do = np.add(-y_actual,y_hat_t)

                ###### compute the new hidden layer derivative #####

                a = np.dot(self.Vy.T, dL_do)

                b= np.dot(self.Wh.T,prev_dh)

                h_next = np.reshape(h_next,len(h_next))

                c = np.diag(np.add(-1,h_next ** 2))

                d = np.dot(c,b)

                #d = np.reshape(np.dot(b,c),(a.shape[0],1))

                dl_dh = np.add(d, a)

                #dL_dW = np.dot(np.dot(np.add(-1, h_t ** 2).T, dl_dh_t), h_t_1.T)
                dL_dW = self.delta_LW(h_current,dl_dh,h_prev)

                #dL_dU = np.dot(np.dot(np.add(-1, h_t ** 2).T, dl_dh_t), input)
                dL_dU = self.delta_LW(h_current,dl_dh,input)

                dW += dL_dW

                dU += dL_dU

                prev_dh = dl_dh


        return (dV, dW, dU)

        # now backprop from the current time step back through time

        # for a single time step this will not be used

    def l2_loss(self,pred,expected):
        '''
        :param pred: array
        :param expected: array
        :return: sum of squared distance
        '''

        error = np.sum(np.abs((expected - pred)**2))

        return error

    def cross_entropy_loss(self,x,y):

        assert len(x) == len(y)

        pass

    def negative_log_loss(self,y_predict,y_expected):
        '''
        take in the predicted y^ which are probabilities of p(y|x1,x2....xn)
        :param y_predict:
        :return: array
        '''

        expected_index = np.where(y_expected == 1)[0][0]
        return -np.log(y_predict[expected_index])

    def delta_LW(self,hidden,dL_dh,input):

        hidden = np.reshape(hidden,len(hidden))

        #a = np.add(-1, hidden ** 2)

        #b = np.diag(a)

        #c = np.dot(b,dL_dh)

        #d = np.dot(c,input.T)

        dLW_dh = np.dot(np.dot(np.diag(np.add(-1, hidden ** 2)), dL_dh), input.T)

        return dLW_dh