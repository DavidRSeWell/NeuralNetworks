import numpy as np
import sys

sys.path.append('/Users/befeltingu/NeuralNetworks/RNN/vanilla_rnn')
sys.path.append('/Users/befeltingu/NeuralNetworks')

from rnn_cell import RNNCell
import utils



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

            o_t = RNN.forward_pass(X[i])

            y_hat = utils.softmax(o_t)

            loss = self.loss(Y[i],y_hat,type)

            tots_loss += loss

        return tots_loss

    def build_transition_p(self,V,W,U):

        t_p_matrix = []

        RNN = RNNCell(Vy=V,Wh=W,Ux=U,h_dim=W.shape[1])

        for word in self.X:

            o_t = RNN.forward_pass(word)

            y_hat = utils.softmax(o_t)

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

            o_t = RNN.forward_pass(self.X[i])

            y_hat = utils.softmax(o_t)

            pred_index = np.argmax(y_hat)

            y_actual_index = np.argmax(self.Y[i])

            if pred_index == y_actual_index:
                correct += 1

        print "accuracy: " + str(correct / float(len(self.X)))
