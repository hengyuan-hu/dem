import keras
from keras.layers import Input
from keras.layers import Convolution2D as Conv2D
from keras.layers import Deconvolution2D as Deconv2D
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils.visualize_util import plot
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10

import os
import cPickle
import tensorflow as tf
import numpy as np
from res_auto import res_autoencoder
import keras_utils
import utils


def lr_schedule(epoch):
    if epoch < 80:
        return 0.1
    elif epoch < 120:
        return 0.01
    else:
        return 0.001


def basic_model(input_shape):
    input_img = Input(shape=input_shape)
    encoded = Conv2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
    decoded = Conv2D(3, 3, 3, activation='linear', border_mode='same')(encoded)
    autoencoder = Model(input_img, decoded)
    return autoencoder


def deep_encoder1(input_shape):
    input_img = Input(shape=input_shape)
    print input_img._keras_shape

    x = Conv2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
    x = BatchNormalization(mode=2, axis=3)(x)
    x = Conv2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = BatchNormalization(mode=2, axis=3)(x)

    x = Conv2D(64, 3, 3, activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    x = Conv2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = BatchNormalization(mode=2, axis=3)(x)

    x = Conv2D(32, 3, 3, activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    x = Conv2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = BatchNormalization(mode=2, axis=3)(x)

    x = Conv2D(10, 1, 1, activation='sigmoid', border_mode='same')(x)

    encoded = x
    return Model(input_img, encoded)


def deep_decoder1(input_shape):
    encoded = Input(shape=input_shape)
    print 'decoder input shape:', encoded._keras_shape

    batch_size = tf.shape(encoded)[0]
    x = BatchNormalization(mode=2, axis=3)(encoded)

    h, w, _ = encoded._keras_shape[1:]
    x = Deconv2D(32, 1, 1, output_shape=[batch_size, h, w, 32],
                 activation='relu', border_mode='same')(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    x = Deconv2D(32, 3, 3, output_shape=[batch_size, h, w, 32],
                 activation='relu', border_mode='same')(x)
    x = BatchNormalization(mode=2, axis=3)(x)

    h *= 2; w *= 2
    x = Deconv2D(64, 3, 3, output_shape=(batch_size, h, w, 64),
                 activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)

    x = Deconv2D(64, 3, 3, output_shape=(batch_size, h, w, 64),
                 activation='relu', border_mode='same', subsample=(1, 1))(x)
    x = BatchNormalization(mode=2, axis=3)(x)

    h *= 2; w *= 2
    x = Deconv2D(32, 3, 3, output_shape=(batch_size, h, w, 32),
                 activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    x = Deconv2D(32, 3, 3, output_shape=(batch_size, h, w, 32),
                 activation='relu', border_mode='same', subsample=(1, 1))(x)
    x = BatchNormalization(mode=2, axis=3)(x)

    x = Deconv2D(3, 3, 3, output_shape=(batch_size, 32, 32, 3),
                 activation='linear', border_mode='same', subsample=(1, 1))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    decoded = x
    return Model(encoded, decoded)


def deep_model1(input_shape):
    input_img = Input(shape=input_shape)
    print 'input shape:', input_img._keras_shape

    x = Conv2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
    x = BatchNormalization(mode=2, axis=3)(x)
    x = Conv2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = BatchNormalization(mode=2, axis=3)(x)

    x = Conv2D(64, 3, 3, activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    x = Conv2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = BatchNormalization(mode=2, axis=3)(x)

    x = Conv2D(32, 3, 3, activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    x = Conv2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = BatchNormalization(mode=2, axis=3)(x)

    x = Conv2D(10, 1, 1, activation='sigmoid', border_mode='same')(x)
    encoded = x
    print 'encoded shape:',  encoded._keras_shape

    batch_size = tf.shape(encoded)[0]
    encoded = GaussianNoise(0.2)(encoded)

    x = BatchNormalization(mode=2, axis=3)(encoded)

    h, w, _ = encoded._keras_shape[1:]
    x = Deconv2D(32, 1, 1, output_shape=[batch_size, h, w, 32],
                 activation='relu', border_mode='same')(x)
    x = BatchNormalization(mode=2, axis=3)(x)

    x = Deconv2D(32, 3, 3, output_shape=[batch_size, h, w, 32],
                 activation='relu', border_mode='same')(x)
    x = BatchNormalization(mode=2, axis=3)(x)

    h *= 2; w *= 2
    x = Deconv2D(64, 3, 3, output_shape=(batch_size, h, w, 64),
                 activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)

    x = Deconv2D(64, 3, 3, output_shape=(batch_size, h, w, 64),
                 activation='relu', border_mode='same', subsample=(1, 1))(x)
    x = BatchNormalization(mode=2, axis=3)(x)

    h *= 2; w *= 2
    x = Deconv2D(32, 3, 3, output_shape=(batch_size, h, w, 32),
                 activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)

    x = Deconv2D(32, 3, 3, output_shape=(batch_size, h, w, 32),
                 activation='relu', border_mode='same', subsample=(1, 1))(x)
    x = BatchNormalization(mode=2, axis=3)(x)

    x = Deconv2D(3, 3, 3, output_shape=(batch_size, 32, 32, 3),
                 activation='linear', border_mode='same', subsample=(1, 1))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    decoded = x
    print 'decoded shape:', decoded._keras_shape
    autoencoder = Model(input_img, decoded)
    return autoencoder


if __name__ == '__main__':
    keras.backend.set_session(utils.get_session())

    (train_xs, _), (test_xs, _) = cifar10.load_data()
    train_xs, mean, std = utils.preprocess_cifar10(train_xs)
    test_xs, _, _ = utils.preprocess_cifar10(test_xs)

    batch_size = 128
    auto = deep_model1(train_xs.shape[1:])
    # auto = res_autoencoder(train_xs.shape[1:], 2)
    opt = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
    auto.compile(optimizer=opt, loss='mse')

    model_name = 'noise_02_deep_model1'
    if not os.path.exists(model_name):
        os.makedirs(model_name)
        plot(auto, to_file=os.path.join(model_name, 'graph.png'), show_shapes=True)
        print 'model graph plotted'

    lrate = LearningRateScheduler(lr_schedule)
    callbacks_list = [lrate]
    auto.fit(train_xs, train_xs,
             nb_epoch=160,
             batch_size=batch_size,
             validation_data=(test_xs, test_xs),
             shuffle=True, callbacks=callbacks_list)
    # model_name = 'noise_deep_model1'
    keras_utils.save_model(auto, os.path.join(model_name, 'encoder_decoder'))
    print 'model saved'

    encoder = deep_encoder1(train_xs.shape[1:])
    decoder = deep_decoder1(encoder.get_output_shape_at(-1)[1:])
    weights = auto.get_weights()
    encoder_weights = encoder.get_weights()
    encoder.set_weights(weights[:len(encoder_weights)])
    decoder.set_weights(weights[len(encoder_weights):])
    keras_utils.save_model(encoder, os.path.join(model_name, 'encoder'))
    print 'encoder saved'
    keras_utils.save_model(decoder, os.path.join(model_name, 'decoder'))
    print 'decoder saved'
