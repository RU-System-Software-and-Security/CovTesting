from __future__ import absolute_import
from __future__ import print_function

import argparse
import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import keras.backend as K
from keras.datasets import mnist, cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.regularizers import l2

import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

####for solving some specific problems, don't care
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def get_model(dataset, softmax=False):
    if dataset == 'svhn1':
        ##MNIST model: 0, 2, 7, 10
        layers = [
            Conv2D(64, (3, 3), padding='valid', input_shape=(32, 32, 3)),  # 0
            Activation('relu'),  # 1
            BatchNormalization(), # 2
            Conv2D(64, (3, 3)),  # 3
            Activation('relu'),  # 4
            BatchNormalization(), # 5
            MaxPooling2D(pool_size=(2, 2)),  # 6
            Dropout(0.5),  # 7
            Flatten(),  # 8
            Dense(128),  # 9
            Activation('relu'),  # 10
            BatchNormalization(), # 11
            Dropout(0.5),  # 12
            Dense(10),  # 13
        ]

    elif dataset == 'svhn2':
        # CIFAR-10 model
        layers = [
            Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),  # 0
            Activation('relu'),  # 1
            BatchNormalization(),  # 2
            Conv2D(32, (3, 3), padding='same'),  # 3
            Activation('relu'),  # 4
            BatchNormalization(),  # 5
            MaxPooling2D(pool_size=(2, 2)),  # 6

            Conv2D(64, (3, 3), padding='same'),  # 7
            Activation('relu'),  # 8
            BatchNormalization(),  # 9
            Conv2D(64, (3, 3), padding='same'),  # 10
            Activation('relu'),  # 11
            BatchNormalization(),  # 12
            MaxPooling2D(pool_size=(2, 2)),  # 13

            Conv2D(128, (3, 3), padding='same'),  # 14
            Activation('relu'),  # 15
            BatchNormalization(),  # 16
            Conv2D(128, (3, 3), padding='same'),  # 17
            Activation('relu'),  # 18
            BatchNormalization(),  # 19
            MaxPooling2D(pool_size=(2, 2)),  # 20

            Flatten(),  # 21
            Dropout(0.5),  # 22

            Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),  # 23
            Activation('relu'),  # 24
            BatchNormalization(),  # 25
            Dropout(0.5),  # 26
            Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),  # 27
            Activation('relu'),  # 28
            BatchNormalization(),  # 29
            Dropout(0.5),  # 30
            Dense(10),  # 31
        ]
    else:
        # SVHN model
        layers = [
            Conv2D(64, (3, 3), padding='valid', input_shape=(32, 32, 3)),  # 0
            Activation('relu'),  # 1
            BatchNormalization(),  # 2
            Conv2D(64, (3, 3)),  # 3
            Activation('relu'),  # 4
            BatchNormalization(),  # 5
            MaxPooling2D(pool_size=(2, 2)),  # 6

            Dropout(0.5),  # 7
            Flatten(),  # 8

            Dense(512),  # 9
            Activation('relu'),  # 10
            BatchNormalization(),  # 11
            Dropout(0.5),  # 12

            Dense(128),  # 13
            Activation('relu'),  # 14
            BatchNormalization(),  # 15
            Dropout(0.5),  # 16
            Dense(10),  # 17
        ]

    model = Sequential()
    for layer in layers:
        model.add(layer)
    if softmax:
        model.add(Activation('softmax'))

    return model

def load_data(name):
    assert (name.upper() in ['MNIST', 'CIFAR', 'SVHN'])
    name = name.lower()
    x_train = np.load('./data/' + name + '_data/' + name + '_x_train.npy')
    y_train = np.load('./data/' + name + '_data/' + name + '_y_train.npy')
    x_test = np.load('./data/' + name + '_data/' + name + '_x_test.npy')
    y_test = np.load('./data/' + name + '_data/' + name + '_y_test.npy')
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    model = get_model('svhn1', softmax=True)
    adam = keras.optimizers.Adam(lr=0.00001)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])  
    model.summary()

    x_train, y_train, x_test, y_test = load_data('svhn')

    # training without data augmentation
    batch_size = 256
    epochs = 100

    model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    # Evaluate the adversarially trained model on clean test set
    labels_true = np.argmax(y_test, axis=1)
    labels_test = np.argmax(model.predict(x_test), axis=1)
    print('Accuracy test set: %.2f%%' % (np.sum(labels_test == labels_true) / x_test.shape[0] * 100))

    model.save('data/svhn_data/model/%s_first.h5' % 'svhn')




