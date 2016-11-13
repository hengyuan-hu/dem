import numpy as np
np.random.seed(0)

import keras
from keras.layers import Input, Dense, MaxPooling2D, UpSampling2D
from keras.layers import Convolution2D as Conv2D
from keras.layers import Deconvolution2D as Deconv2D
from keras.models import Model
from keras.callbacks import TensorBoard
import utils
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras.utils.visualize_util import plot
import os


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
    print 'encoded shape:',  encoded._keras_shape
    # encoder = Model(input_img, encoded)

    # decoder_input = Input(shape=encoded._keras_shape[1:])
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
    print 'decoded shape:', decoded._keras_shape
    autoencoder = Model(input_img, decoded)
    return autoencoder


def save_model(model, model_name):
    model_file, weight_file = _get_model_files(model_name)
    model_json = model.to_json()
    with open(model_file, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(weight_file)
    print('Saved model to disk')


def _get_model_files(model_name):
    model_file = '%s.json' % model_name
    weight_file = '%s.h5' % model_name
    return model_file, weight_file


def load_model(model_name, batch_size):
    model_file, weight_file = _get_model_files(model_name)
    with open(model_file, 'r') as json_file:
        model_json = json_file.read()

    model_json = model_json.replace(
        '"output_shape": [null', '"output_shape": [%d' % batch_size)
    loaded_model = keras.models.model_from_json(model_json)
    loaded_model.load_weights(weight_file)
    print('Loaded model from disk')
    return loaded_model


if __name__ == '__main__':
    keras.backend.set_session(utils.get_session())

    (train_xs, _), (test_xs, _) = cifar10.load_data()
    train_xs, mean, std = utils.preprocess_cifar10(train_xs)
    test_xs, _, _ = utils.preprocess_cifar10(test_xs)

    batch_size = 128
    auto = deep_model1(train_xs.shape[1:])
    # auto = basic_model(train_xs.shape[1:])
    opt = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
    auto.compile(optimizer=opt, loss='mse')

    auto.fit(train_xs, train_xs,
             nb_epoch=80,
             batch_size=batch_size,
             validation_data=(test_xs, test_xs),
             shuffle=True)

    save_model(auto, 'deep_model1')
    # current_dir = os.path.dirname(os.path.realpath(__file__))
    # model_path = os.path.join(current_dir, 'deep_model1.png')
    # plot(auto, to_file=model_path, show_shapes=True)

    # current_dir = os.path.dirname(os.path.realpath(__file__))
    # model_path = os.path.join(current_dir, 'deep_encoder1.png')
    # plot(encoder, to_file=model_path, show_shapes=True)

    # current_dir = os.path.dirname(os.path.realpath(__file__))
    # model_path = os.path.join(current_dir, 'deep_decoder1.png')
    # plot(decoder, to_file=model_path, show_shapes=True)


    #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
