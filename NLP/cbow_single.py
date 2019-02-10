import numpy as np


def softmax(z):

    return np.exp(z) / np.sum(np.exp(z))

def get_posterior(input_word, w_1):
    a = np.matmul(w_1.T, input_word.T)
    a = np.exp(a)
    a = a / sum(a)
    return a

def get_posterior2(input_word, w_1):
    a = np.matmul(w_1, input_word.T)
    a = np.exp(a)
    a = a / sum(a)
    return a


def get_transition_matrix(W0,W1):
    '''

    :param W0:
    :param W1:
    :return:
    '''

    trans_matrix = np.zeros((W0.shape[0],W0.shape[0]))

    for i in range(W0.shape[0]):
        posterior = get_posterior(W0[i],W1)
        trans_matrix[i,:] = posterior

    return trans_matrix

def logistic_regression_cbow(iters,text,vector_size,use_conditional=False):
    '''
    Following derivations from A-matrix paper
    start with binary logistic regression w/ backpropagation
    Steps:
        0: init x , beta , alpha
            X = T x K - (T = size of vocab and K = vector size)
            Beta = 1 x K
        1: for each word in corpus
            predict = logit(x_t,beta)
            p_hat = sample proportion #y / #T - Says in paper #y / #T but it should be #y / #C where #C is corpus size
            p_hat
            x = x + alpha*(p_hat - predict)*


    :return:
    '''

    text = text.replace(',', '').lower()
    text_list = text.split()
    vocab = sorted(list(set(text_list)))
    vocab_size = len(vocab)
    alpha = 0.05  # learning rate
    X = np.random.random((vocab_size, vector_size))
    Beta = np.random.random((vector_size, vocab_size))

    conditional_prob_mat = np.load("/Users/befeltingu/NeuralNetworks/NLP/data/conditional_prob_mat.npy")

    for _ in range(iters):  # one iteration is a full walk through the tex

        for i in range(len(text_list[:-1])):  # go up to the last word. No wrapping

            w_input = text_list[i]
            w_output = text_list[i + 1]
            input_vocab_index = vocab.index(w_input)
            output_vocab_index = vocab.index(w_output)

            y_observed = np.zeros(len(vocab))

            y_observed[output_vocab_index] = 1

            if use_conditional: # use the conditional prob as the observed value

                y_observed = conditional_prob_mat[input_vocab_index]

            x_i = X[input_vocab_index]

            #x_i = np.reshape(x_i,(1,x_i.shape[0]))

            z = np.matmul(x_i,Beta)

            p_hat_i = softmax(z)

            #E = y_observed - p_hat

            #x_i += x_i + alpha*np.matmul(E,x_i)
            for j in range(vocab_size):

                beta_update = alpha*(y_observed[j] - p_hat_i[j])*x_i.T

                x_update = alpha*(y_observed[j] - p_hat_i[j])*Beta[:,j].T

                Beta[:,j] += beta_update
                X[input_vocab_index] += x_update

    return X,Beta




def create_conditional_prob(text):

    text = text.replace(',', '').lower()
    text_list = text.split()
    vocab = sorted(list(set(text_list)))
    vocab_size = len(vocab)

    count_mat = np.zeros((vocab_size,vocab_size))

    for i in range(len(text_list) - 1):

        word = text_list[i]

        next_word = text_list[i + 1]

        index_next = vocab.index(next_word)

        word_index = vocab.index(word)

        count_mat[word_index][index_next] += int(1)

    next_word = text_list[0]
    word = text_list[-1]
    index_next = vocab.index(next_word)
    word_index = vocab.index(word)
    count_mat[word_index][index_next] += int(1)

    row_sumA = np.matrix([count_mat[i].sum() for i in range(len(count_mat))])

    ones_rowA = np.matrix([1 for i in range(len(count_mat))])  # ones vector for mult

    probA = count_mat / (row_sumA.T * ones_rowA)  # Markov trans probl matrix

    np.save('data/conditional_prob_mat',probA)

def simple_cbow(iters, text, vector_size):
    '''
        run through vocab the number of iters
        for each word in text:
            y = predict_next_word
            get the loss
            update the network
    '''

    text = text.replace(',', '').lower()
    text_list = text.split()
    vocab = sorted(list(set(text_list)))
    vocab_size = len(vocab)
    alpha = 0.05  # learning rate
    W_0 = np.random.random((vocab_size, vector_size))
    W_1 = np.random.random((vector_size, vocab_size))

    for _ in range(iters):  # one iteration is a full walk through the tex

        for i in range(len(text_list[:-1])):  # go up to the last word. No wrapping

            w_input = text_list[i]
            w_output = text_list[i + 1]
            input_vocab_index = vocab.index(w_input)
            output_vocab_index = vocab.index(w_output)
            y_observed = np.zeros(len(vocab))

            y_observed[output_vocab_index] = 1

            w_i = W_0[input_vocab_index]  # input vector

            h = w_i.T  # hidden layer equation 1

            # for each input word we are going to make a prediction
            # for each word in the vocabulary
            y_hat = get_posterior(w_i, W_1)  # equation 2,3 in one step

            dE_du = y_hat - y_observed  # equation 8 also e_j

            # calculate gradient from hidden --- > output or W'
            gradient_vector_w_2 = np.zeros((vector_size,vocab_size))  # gradient for 'input' weights. should be a 1 x vector size
            for j in range(vocab_size):

                e_j = dE_du[j]  # for clarity matching notation to Rong

                dE_dw1 = e_j * h  # equation 11

                gradient_vector_w_2.T[j] = dE_dw1



            # now to update the input weights
            # equation 12 tells us that the change in the error w.r.t
            # a single index of the hidden layer results in the sum of the error
            # for the current output j times the ith index for each word in the
            # output weigths. como? que? makes sense in my mind
            gradient_vector_w = np.zeros(vector_size)  # gradient for output weights. 1 x vector size
            for i_hidden in range(vector_size):  # loop over each index in hidden layer
                dE_dh_i = 0.0
                for j_2 in range(vocab_size):
                    e_j = dE_du[j_2]
                    dE_dh_i += e_j * W_1[i_hidden][j_2]

                gradient_vector_w[i_hidden] = dE_dh_i

            # ok I think we can update now
            w_i += -1.0*alpha*gradient_vector_w

            W_1 += -1.0*alpha*gradient_vector_w_2

    return W_0, W_1


if __name__ == '__main__':

    #text = 'I swear to you gentlemen, that to be overly conscious is a sickness, a real, thorough sickness'

    text = '''
        The cat really jumped
        The cat really ate
        The cat almost jumped
        The cat almost ate
        The dog really jumped
        The dog really ate
        The dog almost jumped
        The dog almost ate
    '''

    run_simple_cbow = 0
    if run_simple_cbow:

        W0,W1 = simple_cbow(500, text, 7)

        trans_matrix = get_transition_matrix(W0,W1)

        np.save("data/cbow_simple_trans",trans_matrix)
        np.save("data/cbow_W0",W0)
        np.save("data/cbow_W1",W1)

    run_logistinc_reg_simple = 1
    if run_logistinc_reg_simple:

        X,Beta = logistic_regression_cbow(1500, text, 7,False)

        trans_matrix = get_transition_matrix(X, Beta)

        np.save("data/logistic_trans", trans_matrix)
        np.save("data/logit_X", X)
        np.save("data/logit_Beta", Beta)

    run_create_conditional_prob = 0
    if run_create_conditional_prob:

        create_conditional_prob(text)
