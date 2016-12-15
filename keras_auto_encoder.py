import keras
from keras.layers import Input, Activation
from keras.layers import Convolution2D as Conv2D
from keras.layers import Deconvolution2D as Deconv2D
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils.visualize_util import plot
from keras.callbacks import LearningRateScheduler

import os
import cPickle
import tensorflow as tf
import numpy as np
from res_auto import res_autoencoder
import keras_utils
import utils
from cifar10 import Cifar10Wrapper


def generate_lr_schedule(max_epoch):
    def lr_schedule(epoch):
        if epoch < 0.5 * max_epoch:
            return 0.1
        elif epoch < 0.75 * max_epoch:
            return 0.01
        else:
            return 0.001

    return lr_schedule


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


def deep_encoder2(input_shape):
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
    # latent_dim = (1, 1, 1024)
    x = Conv2D(1024, 4, 4, activation='linear',
                    border_mode='same', subsample=(4, 4))(x)
    x = GaussianNoise(0.1)(x)
    encoded = Activation('sigmoid')(x)
    return Model(input_img, encoded)


def deep_decoder2(input_shape):
    encoded = Input(shape=input_shape)
    print 'encoded shape:', encoded.get_shape().as_list()
    x = BatchNormalization(mode=2, axis=3)(encoded)

    # batch_size, h, w, _ = tf.shape(x)
    batch_size  = tf.shape(x)[0]
    # dim: (1, 1, 512)
    x = Deconv2D(512, 4, 4, output_shape=[batch_size, 4, 4, 512],
                 activation='relu', border_mode='same', subsample=(4, 4))(encoded)
    x = BatchNormalization(mode=2, axis=3)(x)
    # (4, 4, 512)
    x = Deconv2D(256, 5, 5, output_shape=[batch_size, 8, 8, 256],
                 activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    # dim: (8, 8, 236)
    # h *= 2; w *= 2
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
    return Model(encoded, decoded)


def deep_model2(input_shape):
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
    # latent_dim = (1, 1, 1024)
    x = Conv2D(1024, 4, 4, activation='linear', border_mode='same', subsample=(4, 4))(x)
    x = GaussianNoise(0.1)(x)
    encoded = Activation('sigmoid')(x)

    print 'encoded shape:', encoded.get_shape().as_list()
    x = BatchNormalization(mode=2, axis=3)(encoded)

    # batch_size, h, w, _ = tf.shape(x)
    batch_size  = tf.shape(x)[0]
    # dim: (1, 1, 512)
    x = Deconv2D(512, 4, 4, output_shape=[batch_size, 4, 4, 512],
                 activation='relu', border_mode='same', subsample=(4, 4))(encoded)
    x = BatchNormalization(mode=2, axis=3)(x)
    # (4, 4, 512)
    x = Deconv2D(256, 5, 5, output_shape=[batch_size, 8, 8, 256],
                 activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    # dim: (8, 8, 256)
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
    print 'decoded shape:', decoded.get_shape().as_list()
    autoencoder = Model(input_img, decoded)
    return autoencoder


def relu_n(n):
    def clipped_relu(x):
        return keras.activations.relu(x, max_value=n)

    return clipped_relu


def relu_encoder1(input_shape, relu_max):
    input_img = Input(shape=input_shape)
    print 'input shape:', input_img._keras_shape
    # 32, 32
    x = Conv2D(64, 3, 3, activation='relu', border_mode='same', subsample=(2, 2))(input_img)
    x = BatchNormalization(mode=2, axis=3)(x)
    # 16, 16
    x = Conv2D(128, 3, 3, activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    # 8, 8
    x = Conv2D(256, 3, 3, activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    # 4, 4
    # latent_dim = (1, 1, 1024)
    x = Conv2D(1024, 4, 4, activation='linear',
               border_mode='same', subsample=(4, 4))(x)
    x = GaussianNoise(0.2)(x)
    encoded = Activation(relu_n(relu_max))(x)
    return Model(input_img, encoded)


relu_decoder1 = deep_decoder2


def relu_deep_model1(input_shape, relu_max):
    input_img = Input(shape=input_shape)
    print 'input shape:', input_img._keras_shape
    # 32, 32
    x = Conv2D(64, 3, 3, activation='relu', border_mode='same', subsample=(2, 2))(input_img)
    x = BatchNormalization(mode=2, axis=3)(x)
    # 16, 16
    x = Conv2D(128, 3, 3, activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    # 8, 8
    x = Conv2D(256, 3, 3, activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    # 4, 4
    # latent_dim = (1, 1, 1024)
    x = Conv2D(1024, 4, 4, activation='linear',
               border_mode='same', subsample=(4, 4))(x)
    x = GaussianNoise(0.2)(x)
    # encoded = Activation('relu')(x)
    encoded = Activation(relu_n(relu_max))(x)

    print 'encoded shape:', encoded.get_shape().as_list()
    x = BatchNormalization(mode=2, axis=3)(encoded)

    # batch_size, h, w, _ = tf.shape(x)
    batch_size  = tf.shape(x)[0]
    # dim: (1, 1, 512)
    x = Deconv2D(512, 4, 4, output_shape=[batch_size, 4, 4, 512],
                 activation='relu', border_mode='same', subsample=(4, 4))(encoded)
    x = BatchNormalization(mode=2, axis=3)(x)
    # (4, 4, 512)
    x = Deconv2D(256, 5, 5, output_shape=[batch_size, 8, 8, 256],
                 activation='relu', border_mode='same', subsample=(2, 2))(x)
    x = BatchNormalization(mode=2, axis=3)(x)
    # dim: (8, 8, 256)
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
    print 'decoded shape:', decoded.get_shape().as_list()
    autoencoder = Model(input_img, decoded)
    return autoencoder


def test_autoencoder(auto, test_xs):
    init = test_xs[:100]
    utils.vis_cifar10(init, 10, 10, os.path.join(model_name, 'ae_input_imgs.png'))
    pred = auto.predict(init)
    utils.vis_cifar10(pred, 10, 10, os.path.join(model_name, 'ae_recon_imgs.png'))


if __name__ == '__main__':
    keras.backend.set_session(utils.get_session())

    dataset = Cifar10Wrapper()
    train_xs = dataset.train_xs
    test_xs = dataset.test_xs

    batch_size = 128
    num_epoch = 80
    relu_max = 6
    model_name = 'relu_deep_model1_relu_%d' % relu_max

    auto = relu_deep_model1(train_xs.shape[1:], relu_max)
    encoder = relu_encoder1(train_xs.shape[1:], relu_max)
    decoder = relu_decoder1(encoder.get_output_shape_at(-1)[1:])
    opt = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
    auto.compile(optimizer=opt, loss='mse')

    # load pretrained weights
    pretrained_model_name = 'relu_deep_model1'
    weight_file = '%s/encoder_decoder.h5' % pretrained_model_name
    auto.load_weights(weight_file)
    print 'weight loaded from %s' % weight_file

    if not os.path.exists(model_name):
        os.makedirs(model_name)
        plot(auto, to_file=os.path.join(model_name, 'graph.png'), show_shapes=True)
        print 'model graph plotted'

    lrate = LearningRateScheduler(generate_lr_schedule(num_epoch))
    callbacks_list = [lrate]
    auto.fit(train_xs, train_xs,
             nb_epoch=num_epoch,
             batch_size=batch_size,
             validation_data=(test_xs, test_xs),
             shuffle=True, callbacks=callbacks_list)
    keras_utils.save_model(auto, os.path.join(model_name, 'encoder_decoder'))
    print 'model saved'

    weights = auto.get_weights()
    encoder_weights = encoder.get_weights()
    encoder.set_weights(weights[:len(encoder_weights)])
    decoder.set_weights(weights[len(encoder_weights):])
    keras_utils.save_model(encoder, os.path.join(model_name, 'encoder'))
    print 'encoder saved'
    keras_utils.save_model(decoder, os.path.join(model_name, 'decoder'))
    print 'decoder saved'

    test_autoencoder(auto, test_xs)
