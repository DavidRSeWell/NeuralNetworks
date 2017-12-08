# -*- coding: utf-8 -*-


import numpy as np
import utils

import sys

sys.path.append('/Users/befeltingu/NeuralNetworks')
sys.path.append('/Users/befeltingu/NeuralNetworks/RNN/vanilla_rnn')


import utils
from rnn_cell import RNNCell
from rnn_network import NetworkRNN

#######################################
# MY very own RNN from scratch omg ####
#######################################

def softmax(x):

    denom = np.sum(np.exp(x))

    result = np.array([np.exp(x[i])/denom for i in range(len(x))])

    return result


def main():

    np.random.seed(7)

    training_file = '/Users/befeltingu/NeuralNetworks/RNN/data/toy_language'

    training_data = utils.read_data(training_file)

    dictionary, reverse_dictionary,onehot_words = utils.build_dataset(training_data)

    vocab_size = len(dictionary)

    # Parameters
    learning_rate = 0.001
    training_iters = 20000
    display_step = 1000
    window = 2

    # number of units in RNN cell
    h_dimension = 100

    Y = onehot_words[1:]

    X = onehot_words[:-1]

    #h_dimension = 20

    Wh = np.random.random((h_dimension, h_dimension))

    Wh = np.round(Wh,2)

    Ux = np.random.random((h_dimension, X.shape[1]))

    Ux = np.round(Ux,2)

    Vy = np.random.random((X.shape[1], h_dimension))

    Vy = np.round(Vy,2)

    RNNnetwork = NetworkRNN(X, Y)

    RNN = RNNCell(Wh, Ux, Vy, h_dimension, training_data, window=window)

    #learning_rate = 0.001

    loss_array = []

    for d in range(training_iters):

        # the initial hidden layer is all zero

        RNN.h = np.zeros((h_dimension, 1))

        RNN.h_prev = np.zeros((h_dimension, 1))

        RNN.h_next = np.zeros((h_dimension, 1))

        if (d % display_step) == 0:
            # calculate total loss every 5 epochs

            tot_loss = RNNnetwork.total_loss(X, Y, 'negative_log', V=RNN.Vy, W=RNN.Wh, U=RNN.Ux)

            accuracy = RNNnetwork.accuracy(V=RNN.Vy,W=RNN.Wh,U=RNN.Ux)

            loss_array.append(tot_loss)

            print "Iteration: " + str(d)

            print "Total Loss: " + str(tot_loss)

        #for iter in range(len(X)):

        #RNN.h_prev = RNN.h.copy()

        #X_train = X[iter:window + iter]

        dV, dW, dU = RNN.backprop_tt(X, Y)

        RNN.Vy -= learning_rate * dV

        RNN.Ux -= learning_rate * dU

        RNN.Wh -= learning_rate * dW

    return RNNnetwork,RNN,training_data,reverse_dictionary.values()
