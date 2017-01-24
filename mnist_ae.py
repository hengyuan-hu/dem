import tensorflow as tf
import numpy as np
import os
import keras
from keras.layers import Input, Activation, Reshape
from keras.layers import Convolution2D as Conv2D
from keras.layers import Deconvolution2D as Deconv2D
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization as BN


RELU_MAX = 6


def rescale_relu_n(n):
    def f(x):
        return keras.activations.relu(x, max_value=n) / n

    return f


def scale_up(n):
    def f(x):
        return x * n

    return f


def encode(x, use_noise, relu_max=RELU_MAX):
    print 'encode input shape:', x._keras_shape
    assert x._keras_shape[1:] == (28, 28, 1)

    # 28, 28, 1
    y = Conv2D(20, 5, 5, activation='relu', border_mode='same', subsample=(2,2))(x)
    y = BN(mode=2, axis=3)(y)
    # 14, 14, 20
    y = Conv2D(40, 3, 3, activation='relu', border_mode='same', subsample=(2,2))(y)
    y = BN(mode=2, axis=3)(y)

    print 'pre_fc shape:', y._keras_shape
    # 7, 7, 40
    latent_dim = 80
    y = Conv2D(latent_dim, 7, 7, activation='linear',
               border_mode='same', subsample=(7,7))(y)
    # 1, 1, 80
    if use_noise:
        y = GaussianNoise(0.2)(y)
    y = Activation(rescale_relu_n(np.float32(relu_max)))(y)
    y = Reshape((latent_dim,))(y)
    # 80
    return y


def decode(y, relu_max=RELU_MAX):
    # 80
    latent_dim = y._keras_shape[-1]
    x = Reshape((1, 1, latent_dim))(y)
    # 1, 1, 80
    x = Activation(scale_up(np.float32(relu_max)))(x)
    x = BN(mode=2, axis=3)(x)
    batch_size = tf.shape(x)[0]
    x = Deconv2D(40, 7, 7, output_shape=[batch_size, 7, 7, 40], activation='relu',
                 border_mode='same', subsample=(7,7))(x)
    x = BN(mode=2, axis=3)(x)
    # 7, 7, 40
    x = Deconv2D(20, 3, 3, output_shape=[batch_size, 14, 14, 20], activation='relu',
                 border_mode='same', subsample=(2,2))(x)
    x = BN(mode=2, axis=3)(x)
    # 14, 14, 20
    x = Deconv2D(1, 3, 3, output_shape=[batch_size, 28, 28, 1], activation='sigmoid',
                 border_mode='same', subsample=(2,2))(x)
    # 28, 28, 1
    return x


if __name__ == '__main__':
    import keras.backend as K
    from autoencoder import AutoEncoder
    from dataset_wrapper import MnistWrapper
    import utils

    K.set_session(utils.create_session())
    mnist_dataset = MnistWrapper.load_default()
    ae = AutoEncoder(mnist_dataset, encode, decode, 'test/test_mnist_ae')
    # ae.build_models()
    ae.build_models('test/test_mnist_ae')

    # num_epoch = 2
    # lr_schedule = utils.generate_decay_lr_schedule(num_epoch, 0.1, 1)
    # ae.train(128, num_epoch, lr_schedule)
    # ae.save_models()
    # ae.test_models(utils.vis_mnist)
    # ae.log()

    # encoded_dataset = ae.encode(MnistWrapper)
    # encoded_dataset.dump_to_h5('test/test_mnist_ae/encoded_mnist.h5')
