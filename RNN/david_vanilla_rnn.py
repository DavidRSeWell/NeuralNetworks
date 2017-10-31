# -*- coding: utf-8 -*-


import numpy as np
import utils

import sys

sys.path.append('/Users/befeltingu/NeuralNetworks')

import utils

#######################################
# MY very own RNN from scratch omg ####
#######################################

def softmax(x):

    denom = np.sum(np.exp(x))

    result = np.array([np.exp(x[i])/denom for i in range(len(x))])

    return result

class NetworkRNN():

    def __init__(self,x,y):

        self.X = x
        self.Y = y

    def init_cell(self,Wh=None,Ux=None,Vy=None):

        pass

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

    def l2_loss(self, pred, expected):
        '''
        :param pred: array
        :param expected: array
        :return: sum of squared distance
        '''

        error = np.sum(np.abs((expected - pred) ** 2))

        return error

    def loss(self,y,y_hat,type):

        if type == 'cross_entropy':

            return self.cross_entropy_loss(y,y_hat)

        elif type == 'negative_log':

            return self.negative_log_loss(y_predict=y_hat,y_expected=y)

        else:

            print 'using l2 norm - What is this 1989 or something?'

            return self.l2_loss(y,y_hat)

    def total_loss(self,X,Y,type,V,W,U):

        RNN = RNNCell(Vy=V,Wh=W,Ux=U,h_dim=W.shape[1])

        tots_loss = 0

        for i in range(len(Y)):

            y_hat, hidden_t_1, self.h = RNN.forward_pass(X[i],RNN.h)

            loss = self.loss(Y[i],y_hat,type)

            tots_loss += loss

        return tots_loss

    def build_transition_p(self,V,W,U):

        t_p_matrix = []

        RNN = RNNCell(Vy=V,Wh=W,Ux=U,h_dim=W.shape[1])

        for word in self.X:

            y_hat, hidden_t_1, self.h = RNN.forward_pass(word,RNN.h)

            #y_hat = softmax(o_t)

            t_p_matrix.append(y_hat)

        return np.array(t_p_matrix)

    def write_predictions_to_file(self,onehot_words,words,V,W,U,path):

        f_write = open(path,'w')

        predictions = []

        RNN = RNNCell(Vy=V,Wh=W,Ux=U,h_dim=W.shape[1])

        for i in range(len(self.x)):

            y_hat, hidden_t_1, self.h = RNN.forward_pass(self.x[i],RNN.h)

            f_write.write(str(words[np.argmax(y_hat)]) + " ")

        f_write.close()

    def accuracy(self,V,W,U):

        predictions = []

        RNN = RNNCell(Vy=V, Wh=W, Ux=U, h_dim=W.shape[1])

        correct = 0

        for i in range(len(self.X)):

            y_hat, hidden_t_1, self.h = RNN.forward_pass(self.X[i], RNN.h)

            pred_index = np.argmax(y_hat)

            y_actual_index = np.argmax(self.Y[i])

            if pred_index == y_actual_index:
                correct += 1

        print "accuracy: " + str(correct / float(len(self.X)))



class RNNCell():

    def __init__(self,Wh,Ux,Vy,h_dim,words=None,window=1):

        self.Wh = Wh

        self.Ux = Ux

        self.Vy = Vy

        self.h = np.zeros((h_dim,1))

        self.window = window

        self.words = words

    def forward_pass(self,X,hidden_t_1):

        # a(t) = b + Wh(t−1)+ Ux(t)j - activation layer

        X = np.reshape(X,(len(X),1))

        a1 = np.dot(self.Wh,hidden_t_1)

        a2 = np.dot(self.Ux,X)

        a = np.add(a1,a2)

        # h(t) = tanh(a(t)

        self.h = np.tanh(a)

        # o(t) = c + Vh(t)

        o_t = np.dot(self.Vy,self.h)

        # y(t)= softmax(o(t))

        y_hat = softmax(o_t)

        return (y_hat, hidden_t_1 ,self.h)

    def backprop_tt(self,X,Y,t,hidden_t_1):

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

        input = np.reshape(X[t],(len(X[t]),1))
        # dL/do - derivative of Loss w.r.t output

        y_hat_final,h_t_1,h_t = self.forward_pass(X[t],hidden_t_1) # output from the final layer

        y_actual = np.reshape(Y[t],(y_hat_final.shape))

        dL_do = np.add(-y_actual,y_hat_final)

        # dL_dh = derevative of Loss w.r.t hidden layer

        # at the last time step the derivative only depends on the output

        dl_dh_t = np.dot(self.Vy.T,dL_do)

        dL_dV = np.dot(dL_do,h_t.T)

        #dL_dW = np.dot(np.dot(np.add(-1,h_t**2).T,dl_dh_t),h_t_1.T)

        dL_dW = self.delta_LW(h_t,dl_dh_t,h_t_1)

        #dL_dU = np.dot(np.dot(np.add(-1,h_t**2).T,dl_dh_t),input)

        dL_dU = self.delta_LW(h_t,dl_dh_t,input)

        dV += dL_dV

        dW += dL_dW

        dU += dL_dU

        prev_dh = dl_dh_t

        h_tp1 = h_t

        for i in range(t - 1, max(-1,t - self.window), -1):

            input = np.reshape(X[i], (len(X[i]),1))
            # dL/do - derivative of Loss w.r.t output

            y_hat_t, h_t_1, h_t = self.forward_pass(X[i],h_t_1)  # output from the final layer

            dL_do = y_hat_t - 1

            a = np.dot(self.Vy.T, dL_do)

            b= np.dot(self.Wh,prev_dh)

            c = np.diag(np.add(-1,h_tp1 ** 2))

            d = np.reshape(np.dot(b,c),(a.shape[0],1))

            dl_dh_t = np.add(a, d)

            #dL_dW = np.dot(np.dot(np.add(-1, h_t ** 2).T, dl_dh_t), h_t_1.T)
            dL_dW = self.delta_LW(h_t,dl_dh_t,h_t_1)

            #dL_dU = np.dot(np.dot(np.add(-1, h_t ** 2).T, dl_dh_t), input)
            dL_dU = self.delta_LW(h_t,dl_dh_t,input)

            dW += dL_dW

            dU += dL_dU

            prev_dh = dl_dh_t

            h_tp1 = h_t



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

        a = np.add(-1, hidden ** 2)

        b = np.diag(a)

        c = np.dot(b,dL_dh)

        d = np.dot(c,input.T)

        dLW_dh = np.dot(np.dot(np.diag(np.add(-1, hidden ** 2)), dL_dh), input.T)

        return dLW_dh


def main():

    training_file = '/Users/befeltingu/NeuralNetworks/RNN/data/toy_language'

    training_data = utils.read_data(training_file)

    dictionary, reverse_dictionary,onehot_words = utils.build_dataset(training_data)

    vocab_size = len(dictionary)

    # Parameters
    learning_rate = 0.001
    training_iters = 150000
    display_step = 1000
    window = 1

    # number of units in RNN cell
    h_dimension = 512

    Y = onehot_words[1:]

    X = onehot_words[:-1]

    #h_dimension = 20

    Wh = np.random.random((h_dimension, h_dimension))

    Ux = np.random.random((h_dimension, X.shape[1]))

    Vy = np.random.random((X.shape[1], h_dimension))

    RNNnetwork = NetworkRNN(X, Y)

    RNN = RNNCell(Wh, Ux, Vy, h_dimension, training_data, window=window)

    #learning_rate = 0.001

    loss_array = []

    for d in range(training_iters):

        # the initial hidden layer is all zero

        RNN.h = np.zeros((h_dimension, 1))

        if (d % display_step) == 0:
            # calculate total loss every 5 epochs

            tot_loss = RNNnetwork.total_loss(X, Y, 'negative_log', V=RNN.Vy, W=RNN.Wh, U=RNN.Ux)

            accuracy = RNNnetwork.accuracy(V=RNN.Vy,W=RNN.Wh,U=RNN.Ux)

            loss_array.append(tot_loss)

            print "Iteration: " + str(d)

            print "Total Loss: " + str(tot_loss)

        for iter in range(len(X)):

            h_prev = RNN.h

            # utils.save_matrix_img(Wh,'/Users/befeltingu/ML_research/rnn_figures/test.png')

            # (h_hat,h_prev,h_new) = RNN.forward_pass(X[iter],h_prev)

            # error = RNN.negative_log_loss(h_hat,Y[iter])

            dV, dW, dU = RNN.backprop_tt(X, Y, iter, h_prev)

            RNN.Vy -= learning_rate * dV

            RNN.Ux -= learning_rate * dU

            RNN.Wh -= learning_rate * dW

    return RNNnetwork,RNN,training_data,reverse_dictionary.values()