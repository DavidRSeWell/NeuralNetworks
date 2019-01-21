######################################
# utility function used in conjunction
# with various NN
######################################

import numpy as np
import collections
import operator

def softmax(X):

    '''
    simple softmax function implemented in numpy

    :param X:
    :return:
    '''

    X = np.array(X)

    #row_sum = np.array([np.sum(np.exp(r)) for r in X])

    #row_sum = np.reshape(row_sum,(len(row_sum),1))

    return np.exp(X) / np.sum(np.exp(X))

def sigmoid(X):
    return 1/ (1 + np.exp(-X))

def negative_log_loss(y_predict, y_expected):
    '''
    take in the predicted y^ which are probabilities of p(y|x1,x2....xn)
    :param y_predict:
    :return: array
    '''

    expected_index = np.where(y_expected == 1)[0][0]
    return -np.log(y_predict[expected_index])

def plot_matrix(X,x_labels,y_labels,savepath):




    import  matplotlib.pyplot as plt

    '''
    Saves an image of the matrix with
    the labels used as correlations
    if provided
    :param X: numpy array
    :param labels: list
    :return:
    '''

    try:
        assert(type(X) == np.ndarray)

    except Exception,e:
        print "Input matrix must be a numpy array"


    plt.matshow(X)

    x_pos_labels = np.arange(len(x_labels))

    plt.xticks(x_pos_labels,x_labels)

    y_pos = np.arange(len(y_labels))

    plt.yticks(y_pos,y_labels)

    plt.show()

def make_heatmap(X,x_labels,y_labels,savepath):

    '''
     Saves a png of the heatmap of a matrix
    :param X:
    :param x_labels:
    :param y_labels:
    :param savepath:
    :return:
    '''

    import seaborn as sns

    sns.set()

    import pandas as pd

    x_df = pd.DataFrame(X)

    heatmap = sns.heatmap(x_df,xticklabels=x_labels,yticklabels=y_labels,linewidths=1,annot=True, fmt="d")

    heatmap.savefig(savepath)

def read_data(fname):

    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [x.rstrip('\n') for x in content]
    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content

def build_dataset(words):

    '''

    method taken from https://github.com/roatienza/Deep-Learning-Experiments/tree/master/Experiments/Tensorflow/RNN

    :param words:
    :return:
    '''

    '''count = collections.Counter(words).most_common()

    dictionary = dict()

    for word, _ in count:

        dictionary[word] = len(dictionary)'''


    dictionary = dict()

    j = 0
    for i in range(len(words)):

        if words[i] in dictionary:
            continue

        else:

            dictionary[words[i]] = j
            j += 1

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    one_hot_vocab = np.zeros((len(words),len(set(words))))

    for i in range(len((words))):

        one_hot_vocab[i][dictionary[words[i]]] = 1


    return dictionary, reverse_dictionary, one_hot_vocab

def gradient_check(loss):

    '''
    function for checking whether the gradient of a
    function is working properly or not using newton method

    f'(x) = (f(x + h) - f(x - h)) / 2h
    :return:
    '''

def plot_weights(weights,save_path,columns,rows):

    import matplotlib.pyplot as plt

    #weights_y_size = weights.shape[0]
    #weights_x_size = weights.shape[1]

    #fig = plt.figure(figsize=(rows * weights_y_size, columns * weights_x_size))
    fig = plt.figure(figsize=(8 * 5, 4 * 5))

    columns = 4

    rows = 8

    for i in range(1, columns * rows + 1):
        # img = np.random.randint(10, size=(h, w))

        img = weights[0][:, :, 0, i - 1]

        fig.add_subplot(rows, columns, i)

        plt.imshow(img, cmap='gray', interpolation='nearest', aspect='auto')

    fig.tight_layout()

    fig.savefig(save_path, dpi=800)




