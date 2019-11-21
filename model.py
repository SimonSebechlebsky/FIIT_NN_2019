import os

import tensorflow.keras as keras
from tensorflow.keras.layers import concatenate, Conv2D, Dense, Flatten, Layer, MaxPooling2D
from tensorflow.keras import Sequential

from data_manipulation import ROOT_DIR

CLASS_COUNT = 120

class Inception(Layer):
    def __init__(self, filters, activation):
        super(Inception, self).__init__()
        self.filters = filters
        self.activation = activation

        self.conv1 = Conv2D(
            filters=filters,
            kernel_size=1,
            padding='same',
            activation=activation)
        self.conv2 = Conv2D(
            filters=filters,
            kernel_size=1,
            padding='same',
            activation=activation)
        self.conv3 = Conv2D(
            filters=filters,
            kernel_size=1,
            padding='same',
            activation=activation)
        self.conv4 = Conv2D(
            filters=filters,
            kernel_size=1,
            padding='same',
            activation=activation)
        self.conv5 = Conv2D(
            filters=filters,
            kernel_size=3,
            padding='same',
            activation=activation)
        self.conv6 = Conv2D(
            filters=filters,
            kernel_size=5,
            padding='same',
            activation=activation)
        self.mp1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')

    def call(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        m = self.mp1(x)
        c5 = self.conv5(c1)
        c6 = self.conv6(c2)
        c4 = self.conv4(m)
        filter_concatenation = concatenate([c3, c4, c5, c6])
        return filter_concatenation

    def get_config(self):
        return {
            'filters': self.filters,
            'activations': self.activation
        }


def get_model():
    model = Sequential()
    model.add(Inception(32, activation='relu'))
    model.add(MaxPooling2D(pool_size=(7, 7)))
    model.add(Inception(64, activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5)))
    model.add(Inception(128, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(CLASS_COUNT, activation='softmax'))
    return model


model = get_model()
