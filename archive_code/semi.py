from dataset_wrapper import Cifar10Wrapper, MnistWrapper

import tensorflow as tf
import numpy as np
import os
import keras
from keras.layers import Input, Activation, Dense, Dropout
from keras.models import Model
from keras.utils import np_utils

# from keras.layers import Convolution2D as Conv2D
# from keras.layers import Deconvolution2D as Deconv2D
# from keras.layers.noise import GaussianNoise
# from keras.layers.normalization import BatchNormalization as BN
import utils


def fc1_model(x_shape):
    x = Input(x_shape)
    # y = Dense(100, activation='relu')(x)
    # y = Dropout(0.5)(y)
    # y = Dense(1024, activation='relu')(x)
    # y = Dropout(0.5)(y)
    y = Dense(10, activation='softmax')(x)
    return Model(x, y)


def semi_supervised(dataset, model):
    opt = keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
    callback_list = []
    # callback_list = [LearningRateScheduler(lr_schedule)]
    num_epoch = 100
    batch_size = 128
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(
        dataset.train_xs, dataset.train_ys,
        nb_epoch=num_epoch,
        batch_size=batch_size,
        validation_data=(dataset.test_xs, dataset.test_ys),
        shuffle=True, callbacks=callback_list)
    # self.history = history.history


if __name__ == '__main__':
    # dataset = Cifar10Wrapper.load_from_h5(
    #     'prod/cifar10_ae2_relu_6/encoded_cifar10.h5')

    # dataset = Cifar10Wrapper.load_default()
    dataset = MnistWrapper.load_from_h5('test/mnist_ae_relu_6/encoded_mnist.h5')
    dataset.reshape((-1,))
    model = fc1_model(dataset.x_shape)

    print dataset.train_xs.shape, dataset.train_ys.shape
    print dataset.test_xs.shape, dataset.test_ys.shape
    # print dataset.train_ys[:10]
    # print dataset.test_ys[:10]

    dataset.train_ys = np_utils.to_categorical(dataset.train_ys)
    dataset.test_ys = np_utils.to_categorical(dataset.test_ys)

    dataset.train_xs = dataset.train_xs[:100]
    dataset.train_ys = dataset.train_ys[:100]

    semi_supervised(dataset, model)
