# -*- coding: utf-8 -*-


import numpy as np
import utils
'''
    The goal of this is just to play around with Dash and also to employ a
    few different techniques to solving a few different ML tasks in regards
    to a simple dice game

    It will allow me to visualize and try out a few different algorithms/ approaches
    to solving the same problem

'''


class DiceNN:

    def __init__(self,hidden_size,input_size):

        self.hidden_size = hidden_size

        self.input_size = input_size

        self.alpha = 0.05

        self.output_size = input_size

        self.h = np.zeros(hidden_size)

        self.W = np.random.random((self.hidden_size,self.input_size)) # input weights

        self.V = np.random.random((self.output_size,self.hidden_size)) # output weights


    def get_random_state(self,size):

        state = np.zeros((1,12))

        dice_roll = np.random.randint(2,13)

        state[dice_roll] = 1

    def create_data_set(self,size):

        for i in range(size):
            state = 0
            dice_roll = np.random.randint(2,13)

    def forward_pass(self,X):

        # a(t) = b + Wx(t)j - activation layer

        X = np.reshape(X,(len(X),1))

        a1 = np.dot(self.W,X)

        # h(t) = tanh(a(t)

        self.h = utils.sigmoid(a1)

        # o(t) = c + Vh(t)

        o_t = np.dot(self.V,self.h)

        # y(t)= softmax(o(t))

        y_hat = utils.softmax(o_t)

        return y_hat

    def backprop(self,x,y_hat):

        # DL / DV = (y^ - 1) * h
        # DL / DW = y^(1 - y^)* V * h(t) * ( 1- h(t))

        dl_dv = np.dot((y_hat - 1),self.h)

        dl_dw_1 = np.dot((y_hat -1),self.V)

        dl_dw_2 = np.dot(dl_dw_1,self.h)

        dl_dw_3 = np.dot(dl_dw_2,(1 - self.h))

        dl_dw_4 = np.dot(dl_dw_3,x)

        self.W += self.alpha*dl_dw_4

        self.V += self.alpha*dl_dv

    def train(self,iterations):

        for i in range(iterations):
            pass



