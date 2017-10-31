######################################
# utility function used in conjunction
# with various NN
######################################

import numpy as np
import collections

def softmax(X):

    '''
    simple softmax function implemented in numpy

    :param X:
    :return:
    '''
    row_sum = np.array([np.sum(np.exp(r)) for r in X])

    row_sum = np.reshape(row_sum,(len(row_sum),1))

    return np.exp(X) / row_sum

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

    count = collections.Counter(words).most_common()

    dictionary = dict()

    for word, _ in count:

        dictionary[word] = len(dictionary)

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    one_hot_vocab = np.zeros((len(words),len(set(words))))

    for i in range(len(words)):

        one_hot_vocab[i][dictionary[words[i]]] = 1


    return dictionary, reverse_dictionary, one_hot_vocab


