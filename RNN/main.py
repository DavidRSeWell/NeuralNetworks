###################################
# THIS IS THE MAIN FILE FOR RUNNING
# DIFFERENT RNN networks
###################################

import sys

sys.path.append('/Users/befeltingu/NeuralNetworks')

import utils


#####################################
#    TABLE OF CONTENTS
#  ---------------------------
# 1 - LSTM implemented by Siraj Raval
######################################


######################################
#       Siraj Raval LSTM
######################################
run_raval_lstm = 0
if run_raval_lstm:

    from raval_lstm import RecurrentNeuralNetwork,LoadText

    import numpy as np

    # Begin program
    print("Beginning")

    iterations = 10000

    learningRate = 0.001

    # load input output data (words)
    returnData, numCategories, expectedOutput, outputSize, data,words = LoadText()

    print("Done Reading")

    # init our RNN using our hyperparams and dataset
    RNN = RecurrentNeuralNetwork(numCategories, numCategories, outputSize, expectedOutput, learningRate)

    # training time!
    for i in range(1, iterations):

        # compute predicted next word
        RNN.forwardProp()

        # update all our weights using our error
        error = RNN.backProp()

        # once our error/loss is small enough

        if (i % 100) == 0:
            print("Error on iteration ", i, ": ", error)




    # Lets create a transition prob matrix to look at

    output_matrix = RNN.oa

    trans_prob_matrix = utils.softmax(output_matrix)

    np.save('/Users/befeltingu/NeuralNetworks/RNN/output/raval_trans_prob.npy',trans_prob_matrix)

    #utils.plot_matrix(trans_prob_matrix,x_labels=data,y_labels=words,savepath="")

######################################
#       Tensorflow LSTM
######################################
run_tensorflow_lstm = 0
if run_tensorflow_lstm:

    import tensorflow_lstm

    import numpy as np

    predict_matrix,training_data,words = tensorflow_lstm.main()

    predict_matrix = np.array(predict_matrix)

    predict_matrix = np.reshape(predict_matrix,(predict_matrix.shape[0],predict_matrix.shape[2]))

    softmax_mat = utils.softmax(predict_matrix)

    np.save('/Users/befeltingu/NeuralNetworks/RNN/output/tensorflow/predictions.npy',predict_matrix)

    np.save('/Users/befeltingu/NeuralNetworks/RNN/output/tensorflow/rnn_prob_trans.npy',softmax_mat)

    np.save('/Users/befeltingu/NeuralNetworks/RNN/output/tensorflow/y_labels.npy',training_data)

    np.save('/Users/befeltingu/NeuralNetworks/RNN/output/tensorflow/x_labels.npy',words)


    print "done with tensorflow lstm example"

######################################
#       David RNN
######################################
run_david_rnn = 1
if run_david_rnn:

    print "runnin davids rnn"

    import david_vanilla_rnn

    import numpy as np

    RNNnetwork,RNN,training_data,words = david_vanilla_rnn.main()

    t_p_matrix = RNNnetwork.build_transition_p(RNN.Vy, RNN.Wh, RNN.Ux)

    t_p_matrix = np.reshape(t_p_matrix, (t_p_matrix.shape[0], t_p_matrix.shape[1]))

    np.save('/Users/befeltingu/NeuralNetworks/RNN/output/david/pmatrix.npy', t_p_matrix)

    np.save('/Users/befeltingu/NeuralNetworks/RNN/output/david/x_labels.npy', words)

    np.save('/Users/befeltingu/NeuralNetworks/RNN/output/david/y_labels.npy', training_data)

######################################
#       Visualize data
######################################
run_visualization = 0
if run_visualization:

    import numpy as np

    run_raval_lstm_vis = 1
    if run_raval_lstm_vis:

        from raval_lstm import RecurrentNeuralNetwork, LoadText

        import sys

        sys.path.append('/Users/befeltingu/NeuralNetworks')

        import utils

        returnData, numCategories, expectedOutput, outputSize, data, words = LoadText()

        prob_trans_matrix = np.load('/Users/befeltingu/NeuralNetworks/RNN/output/raval_trans_prob.npy')

        utils.make_heatmap(prob_trans_matrix[1:],x_labels=data,y_labels=words,savepath='/Users/befeltingu/NeuralNetworks/RNN/output/raval_trans_prob_heatmap')

