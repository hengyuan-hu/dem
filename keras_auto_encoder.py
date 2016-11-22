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
from keras import regularizers
from keras.layers.noise import GaussianNoise
import cPickle
from keras.callbacks import LearningRateScheduler
from res_auto import res_autoencoder


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


def load_encoder_decoder(input_shape, encoder_func,  encoder_name,
                         decoder_func, decoder_name):
    encoder = encoder_func(input_shape)
    decoder = decoder_func(encoder.get_output_shape_at(-1)[1:])

    _, encoder_weight_file = _get_model_files(encoder_name)
    _, decoder_weight_file = _get_model_files(decoder_name)

    encoder.load_weights(encoder_weight_file)
    decoder.load_weights(decoder_weight_file)
    print('Loaded model from disk')
    return encoder, decoder


def load_coder(input_shape, coder_func,  coder_name):
    coder = coder_func(input_shape)
    _, coder_weight_file = _get_model_files(coder_name)
    coder.load_weights(coder_weight_file)
    print 'Loaded %s from disk' % coder_name
    return coder


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
    encoded = GaussianNoise(0.1)(encoded)

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
    # model_json = model.to_json()
    # with open(model_file, 'w') as json_file:
    #     json_file.write(model_json)
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
    # auto = deep_model1(train_xs.shape[1:])
    auto = res_autoencoder(train_xs.shape[1:], 2)
    opt = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
    auto.compile(optimizer=opt, loss='mse')

    model_name = 'res_auto'
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
    save_model(auto, os.path.join(model_name, 'encoder_decoder'))
    print 'model saved'

    # encoder = deep_encoder1(train_xs.shape[1:])
    # decoder = deep_decoder1(encoder.get_output_shape_at(-1)[1:])
    # weights = auto.get_weights()
    # encoder_weights = encoder.get_weights()
    # encoder.set_weights(weights[:len(encoder_weights)])
    # decoder.set_weights(weights[len(encoder_weights):])
    # save_model(encoder, os.path.join(model_name, 'encoder'))
    # print 'encoder saved'
    # save_model(decoder, os.path.join(model_name, 'decoder'))
    # print 'decoder saved'
