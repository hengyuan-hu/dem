import keras
from keras.layers import Input, Lambda
from keras.layers import Convolution2D as Conv2D
from keras.layers import Deconvolution2D as Deconv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.models import Model
from keras.datasets import cifar10
from keras.utils.visualize_util import plot
from keras.callbacks import LearningRateScheduler
import keras.backend as K

import os
import cPickle
import tensorflow as tf
import numpy as np
import utils
import keras_utils


def deep_encoder1(input_shape):
    input_img = Input(shape=input_shape)
    print input_img._keras_shape

    x = Conv2D(32, 3, 3, activation='relu', border_mode='same', subsample=(2, 2))(input_img)
    x = BatchNormalization(mode=2, axis=3)(x)
    # 16, 16
    x = Conv2D(64, 3, 3, activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    # 8, 8
    x = Conv2D(128, 3, 3, activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    # 4, 4
    latent_dim = (1, 1, 1024)
    z_mean = Conv2D(1024, 4, 4, activation='linear',
                    border_mode='same', subsample=(4, 4))(x)
    z_log_var = Conv2D(1024, 4, 4, activation='linear',
                       border_mode='same', subsample=(4, 4))(x)
    encoded = K.concatenate([z_mean, z_log_var], axis=1)
    # print 'encoded_shape', encoded.get_shape().as_list()
    return Model(input_img, [z_mean, z_log_var])


def deep_decoder1(input_shape):
    z = Input(shape=input_shape)
    print 'decoder input shape:', z._keras_shape

    batch_size = tf.shape(z)[0]
    h, w, _ = z._keras_shape[1:]
    # dim: (1, 1, 512)
    x = Deconv2D(512, 4, 4, output_shape=[batch_size, 4, 4, 512],
                 activation='relu', border_mode='same', subsample=(4, 4))(z)
    x = BatchNormalization(mode=2, axis=3)(x)
    # (4, 4, 512)
    x = Deconv2D(256, 5, 5, output_shape=[batch_size, 8, 8, 256],
                 activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    # dim: (8, 8, 236)
    h *= 2; w *= 2
    x = Deconv2D(128, 5, 5, output_shape=(batch_size, 16, 16, 128),
                 activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    # dim: (16, 16, 256)
    x = Deconv2D(64, 5, 5, output_shape=(batch_size, 32, 32, 64),
                 activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    # dim: (32, 32, 64)
    x = Deconv2D(3, 5, 5, output_shape=(batch_size, 32, 32, 3),
                 activation='linear', border_mode='same', subsample=(1, 1))(x)
    decoded = BatchNormalization(mode=2, axis=3)(x)
    return Model(z, decoded)


def sampling_gaussian((mean, log_var)):
    epsilon = K.random_normal(K.shape(mean), 0.0, 1.0)
    return mean + K.exp(log_var / 2) * epsilon


def deep_model1(input_shape):
    input_img = Input(shape=input_shape)
    print 'input shape:', input_img._keras_shape
    # 32, 32
    x = Conv2D(32, 3, 3, activation='relu', border_mode='same', subsample=(2, 2))(input_img)
    x = BatchNormalization(mode=2, axis=3)(x)
    # 16, 16
    x = Conv2D(64, 3, 3, activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    # 8, 8
    x = Conv2D(128, 3, 3, activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    # 4, 4
    latent_dim = (1, 1, 1024)
    z_mean = Conv2D(1024, 4, 4, activation='linear',
                    border_mode='same', subsample=(4, 4))(x)
    # z_mean = GaussianNoise(0.1)(z_mean)
    # TODO: the next layer use 16K parameters, will it be a problem?
    z_log_var = Conv2D(1024, 4, 4, activation='linear',
                       border_mode='same', subsample=(4, 4))(x)
    z = Lambda(sampling_gaussian, output_shape=latent_dim)([z_mean, z_log_var])
    print 'encoded shape:', z._keras_shape
    # x = BatchNormalization(mode=2, axis=3)(z)

    batch_size = tf.shape(z)[0]
    h, w, _ = z._keras_shape[1:]
    # dim: (1, 1, 512)
    x = Deconv2D(512, 4, 4, output_shape=[batch_size, 4, 4, 512],
                 activation='relu', border_mode='same', subsample=(4, 4))(z)
    x = BatchNormalization(mode=2, axis=3)(x)
    # (4, 4, 512)
    x = Deconv2D(256, 5, 5, output_shape=[batch_size, 8, 8, 256],
                 activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    # dim: (8, 8, 236)
    h *= 2; w *= 2
    x = Deconv2D(128, 5, 5, output_shape=(batch_size, 16, 16, 128),
                 activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    # dim: (16, 16, 256)
    x = Deconv2D(64, 5, 5, output_shape=(batch_size, 32, 32, 64),
                 activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    # dim: (32, 32, 64)
    x = Deconv2D(3, 5, 5, output_shape=(batch_size, 32, 32, 3),
                 activation='linear', border_mode='same', subsample=(1, 1))(x)
    decoded = BatchNormalization(mode=2, axis=3)(x)
    print 'decoded shape:', decoded._keras_shape
    autoencoder = Model(input_img, decoded)

    # define vae loss
    def vae_loss(y, y_pred):
        # TODO: generalize this function
        recon_loss = K.sum(K.square(y_pred - y), axis=[1, 2, 3])
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
                               axis=[1, 2, 3])
        print ('pre average loss shape:',
               recon_loss.get_shape().as_list(),
               kl_loss.get_shape().as_list())
        return K.mean(recon_loss + kl_loss)

    return autoencoder, vae_loss


if __name__ == '__main__':
    keras.backend.set_session(utils.get_session())

    (train_xs, _), (test_xs, _) = cifar10.load_data()
    train_xs, mean, std = utils.preprocess_cifar10(train_xs)
    test_xs, _, _ = utils.preprocess_cifar10(test_xs)

    batch_size = 128
    auto, loss = deep_model1(train_xs.shape[1:])
    # opt = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
    opt = keras.optimizers.Adam()
    auto.compile(optimizer=opt, loss=loss)

    model_name = 'vae2'
    if not os.path.exists(model_name):
        os.makedirs(model_name)
    plot(auto, to_file=os.path.join(model_name, 'graph.png'), show_shapes=True)
    print 'model graph plotted'

    # auto.load_weights(os.path.join(model_name, 'encoder_decoder.h5'))
    lrate = LearningRateScheduler(lr_schedule)
    callbacks_list = [lrate]
    auto.fit(train_xs, train_xs,
             nb_epoch=50,
             batch_size=batch_size,
             validation_data=(test_xs, test_xs),
             shuffle=True, callbacks=callbacks_list)
    keras_utils.save_model(auto, os.path.join(model_name, 'encoder_decoder'))
    print 'model saved'

    encoder = deep_encoder1(train_xs.shape[1:])
    decoder = deep_decoder1((1, 1, 1024))
    weights = auto.get_weights()
    encoder_weights = encoder.get_weights()
    encoder.set_weights(weights[:len(encoder_weights)])
    decoder_weights = decoder.get_weights()
    decoder.set_weights(weights[-len(decoder_weights):])
    keras_utils.save_model(encoder, os.path.join(model_name, 'encoder'))
    print 'encoder saved'
    keras_utils.save_model(decoder, os.path.join(model_name, 'decoder'))
    print 'decoder saved'
