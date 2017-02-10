import tensorflow as tf
import numpy as np
import os
import keras
from keras.layers import Input, Activation, Reshape
from keras.layers import Convolution2D as Conv2D
from keras.layers import Deconvolution2D as Deconv2D
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization as BN
import utils


RELU_MAX = 6


def encode(x, use_noise, relu_max):
    print 'encoder input shape:', x._keras_shape
    assert x._keras_shape[1:] == (32, 32, 3)

    # 32, 32, 3
    y = Conv2D(64, 3, 3, activation='relu', border_mode='same', subsample=(2,2))(x)
    y = BN(mode=2, axis=3)(y)
    # 16, 16, 64
    y = Conv2D(128, 3, 3, activation='relu', border_mode='same', subsample=(2,2))(y)
    y = BN(mode=2, axis=3)(y)
    # 8, 8, 128
    y = Conv2D(256, 3, 3, activation='relu', border_mode='same', subsample=(2,2))(y)
    y = BN(mode=2, axis=3)(y)
    # 4, 4, 256
    assert y._keras_shape[1:] == (4, 4, 256), \
        '%s vs %s' % (y._keras_shape[1:], [4, 4, 256])
    latent_dim = 1024
    y = Conv2D(latent_dim, 4, 4, activation='linear',
               border_mode='same', subsample=(4,4))(y)
    # 1, 1, latent_dim
    if use_noise and not relu_max:
        print 'add noise and pretend relu_max will be:', RELU_MAX
        y = GaussianNoise(0.2 * RELU_MAX)(y)

    y = Activation(utils.relu_n(relu_max))(y)
    if relu_max:
        print 'relu_max:', relu_max
        y = Activation(utils.scale_down(relu_max))(y)
        # y in [0, 1]
        # if use_noise:
        #     y = GaussianNoise(0.2)(y)
        #     y = Activation('relu')(y)

    y = Reshape((latent_dim,))(y) # or Reshape([-1])(y) ?
    # latent_dim
    return y


def decode(y, use_noise, relu_max):
    print 'decoder input shape:', y._keras_shape
    assert len(y._keras_shape) == 2
    assert use_noise == bool(relu_max), 'use noise means use relu max.'
    if relu_max and use_noise:
        x = GaussianNoise(0.2)(y)
        x = Activation(utils.relu_n(1))(x)
    else:
        x = y

    latent_dim = x._keras_shape[-1]
    x = Reshape((1, 1, latent_dim))(x)
    # 1, 1, latent_dim
    if relu_max:
        print 'in decode: relu_max:', relu_max
        x = Activation(utils.scale_up(relu_max))(x)
    # x = BN(mode=2, axis=3)(x) # this bn seems not good? NOT VERIFIED

    # why use 512 instead of 256 here?
    batch_size = keras.backend.shape(x)[0]
    x = Deconv2D(512, 4, 4, output_shape=[batch_size, 4, 4, 512],
                 activation='relu', border_mode='same', subsample=(4,4))(x)

    x = BN(mode=2, axis=3)(x)
    # 4, 4, 512
    x = Deconv2D(256, 5, 5, output_shape=[batch_size, 8, 8, 256],
                 activation='relu', border_mode='same', subsample=(2,2))(x)
    x = BN(mode=2, axis=3)(x)
    # 8, 8, 256
    x = Deconv2D(128, 5, 5, output_shape=(batch_size, 16, 16, 128),
                 activation='relu', border_mode='same', subsample=(2,2))(x)
    x = BN(mode=2, axis=3)(x)
    # 16, 16, 256
    x = Deconv2D(64, 5, 5, output_shape=(batch_size, 32, 32, 64),
                 activation='relu', border_mode='same', subsample=(2,2))(x)
    x = BN(mode=2, axis=3)(x)
    # 32, 32, 64
    x = Deconv2D(3, 5, 5, output_shape=(batch_size, 32, 32, 3),
                 activation='linear', border_mode='same', subsample=(1, 1))(x)
    # 32, 32, 3
    x = BN(mode=2, axis=3)(x)
    return x


if __name__ == '__main__':
    import keras.backend as K
    from autoencoder import AutoEncoder
    from dataset_wrapper import Cifar10Wrapper
    import utils

    K.set_session(utils.create_session())
    cifar10_dataset = Cifar10Wrapper.load_default()

    # ----------normal relu pretraining----------
    print 'Training model with normal relu'
    folder = 'prod/archive/cifar10_ae2_relu_inf'
    # ae = AutoEncoder(cifar10_dataset, encode, decode, None, folder)
    # ae.build_models()

    num_epoch = 160
    # no decay
    lr_schedule = utils.generate_decay_lr_schedule(num_epoch, 0.1, 1)
    # ae.train(128, num_epoch, lr_schedule)
    # ae.save_models()
    # ae.test_models(utils.vis_cifar10)
    # ae.log()

    # encoded_dataset = ae.encode(Cifar10Wrapper)
    # encoded_dataset.dump_to_h5(os.path.join(folder, 'encoded_cifar10.h5'))
    # encoded_dataset.plot_data_dist(os.path.join(folder, 'encoded_plot.png'))

    # ----------truncate relu and fine-tune----------
    print 'Training model with relu-%d' % RELU_MAX
    new_folder = 'prod/cifar10_ae3_relu_%d' % RELU_MAX
    ae = AutoEncoder(cifar10_dataset, encode, decode, RELU_MAX, new_folder)
    ae.build_models(folder) # load previously trained ae

    # num_epoch = 2
    # lr_schedule = utils.generate_decay_lr_schedule(num_epoch, 0.1, 1)
    ae.train(128, num_epoch, lr_schedule)
    ae.save_models()
    ae.log()
    ae.test_models(utils.vis_cifar10)

    encoded_dataset = ae.encode(Cifar10Wrapper)
    encoded_dataset.dump_to_h5(os.path.join(new_folder, 'encoded_cifar10.h5'))
    encoded_dataset.plot_data_dist(os.path.join(new_folder, 'encoded_plot.png'))
