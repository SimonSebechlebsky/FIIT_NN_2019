import tensorflow.keras as keras
from tensorflow.keras.layers import concatenate, Conv2D, Dense, Flatten, Layer, MaxPooling2D

class Inception(Layer):
    def __init__(self, filters, activation):
        super(Inception, self).__init__()

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


class InceptionSimpleModel(keras.Model):

    def __init__(self, filters, dim_output):
        super(InceptionSimpleModel, self).__init__()
        self.model_layers = [
            Inception(
                filters=filters,
                activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Inception(
                filters=filters,
                activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Inception(
                filters=filters,
                activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(
                units=512,
                activation='relu'),
            Dense(
                units=dim_output,
                activation='softmax')
        ]

    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x
