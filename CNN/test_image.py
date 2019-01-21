

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PIL import Image
from utils import plot_weights

import numpy as np
import matplotlib.pyplot as plt

run_test_pil = 0
if run_test_pil:

    w, h = 512, 512

    data = np.random.random((h, w))

    img = Image.fromarray(data, '1')

    img.show()

run_view_mnist = 1
if run_view_mnist:

    num_classes = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':

        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)

        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

        input_shape = (1, img_rows, img_cols)
    else:

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')

    x_test = x_test.astype('float32')

    x_train /= 255

    x_test /= 255

    print('x_train shape:', x_train.shape)

    print(x_train.shape[0], 'train samples')

    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices

    y_train = keras.utils.to_categorical(y_train, num_classes)

    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))

    model.add(Conv2D(64, (5, 5), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])



    weights = model.get_weights()

    #plt.imshow(x_train[0][:,:,0],cmap='gray')

    #plt.show()

    w = 10
    h = 10
    fig = plt.figure(figsize=(8*5, 4*5))

    columns = 4
    rows = 8

    plot_weights(weights,'/Users/befeltingu/Documents/PlantMain/test.png',columns,rows)

    '''
    for i in range(1, columns * rows + 1):

        #img = np.random.randint(10, size=(h, w))

        img = weights[0][:,:,0,i - 1]

        fig.add_subplot(rows, columns, i)

        plt.imshow(img,cmap='gray',interpolation='nearest',aspect='auto')

    #fig.tight_layout()

    #fig.subplots_adjust(wspace=0, hspace=0)

    fig.savefig('/Users/befeltingu/NeuralNetworks/CNN/data/test_weight_image.png')
    '''
