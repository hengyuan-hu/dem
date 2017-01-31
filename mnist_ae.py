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
    assert x._keras_shape[1:] == (28, 28, 1)

    # 28, 28, 1
    y = Conv2D(20, 5, 5, activation='relu', border_mode='same', subsample=(2,2))(x)
    y = BN(mode=2, axis=3)(y)
    # 14, 14, 20
    y = Conv2D(40, 3, 3, activation='relu', border_mode='same', subsample=(2,2))(y)
    y = BN(mode=2, axis=3)(y)
    # 7, 7, 40
    print 'pre_fc shape:', y._keras_shape
    latent_dim = 80
    y = Conv2D(latent_dim, 7, 7, activation='linear',
               border_mode='same', subsample=(7,7))(y)
    # 1, 1, latent_dim
    if use_noise and not relu_max:
        print 'add noise and pretend relu_max will be:', RELU_MAX
        y = GaussianNoise(0.2 * RELU_MAX)(y)

    y = Activation(utils.relu_n(relu_max))(y)
    if relu_max:
        print 'relu max:', relu_max
        y = Activation(utils.scale_down(relu_max))(y)
        # y in [0, 1]
        if use_noise:
            y = GaussianNoise(0.2)(y)
            y = Activation('relu')(y)
    y = Reshape((latent_dim,))(y)
    # 80
    return y


def decode(y, relu_max):
    assert len(y._keras_shape) == 2
    latent_dim = y._keras_shape[-1]
    x = Reshape((1, 1, latent_dim))(y)
    # 1, 1, latent_dim
    if relu_max:
        x = Activation(utils.scale_up(relu_max))(x)
    # not good? x = BN(mode=2, axis=3)(x)

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

    # ----------normal relu pretraining----------
    print 'Training model with normal relu'
    folder = 'test/mnist_ae_relu_inf'
    ae = AutoEncoder(mnist_dataset, encode, decode, None, folder)
    ae.build_models()

    num_epoch = 30
    lr_schedule = utils.generate_decay_lr_schedule(num_epoch, 0.1, 1)
    ae.train(128, num_epoch, lr_schedule)
    ae.save_models()
    ae.test_models(utils.vis_mnist)
    ae.log()

    encoded_dataset = ae.encode(MnistWrapper)
    encoded_dataset.dump_to_h5(os.path.join(folder, 'encoded_mnist.h5'))
    encoded_dataset.plot_data_dist(os.path.join(folder, 'encoded_plot.png'))

    # ----------truncate relu and fine-tune----------
    print 'Training model with relu-%d' % RELU_MAX
    new_folder = 'test/mnist_ae_relu_%d' % RELU_MAX
    ae = AutoEncoder(mnist_dataset, encode, decode, RELU_MAX, new_folder)
    ae.build_models(folder) # load previously trained ae

    mnist_dataset.plot_data_dist(os.path.join(new_folder, 'original_mnist_plot.png'))

    # num_epoch = 2
    # lr_schedule = utils.generate_decay_lr_schedule(num_epoch, 0.1, 1)
    ae.train(128, num_epoch, lr_schedule)
    ae.save_models()
    ae.test_models(utils.vis_mnist)
    ae.log()

    encoded_dataset = ae.encode(MnistWrapper)
    encoded_dataset.dump_to_h5(os.path.join(new_folder, 'encoded_mnist.h5'))
    encoded_dataset.plot_data_dist(os.path.join(new_folder, 'encoded_plot.png'))
