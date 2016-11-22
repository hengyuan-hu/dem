import os
import utils
import keras
from keras.models import Model
from keras.layers import Input, Activation, merge, Dense
from keras.layers.convolutional import Convolution2D, Deconvolution2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.noise import GaussianNoise
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import plot

import tensorflow as tf
from keras import backend as K


# Helper to build a conv -> BN -> relu block
# def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
#     def f(x):
#         conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col,
#                              W_regularizer=l2(1e-4), subsample=subsample,
#                              init="he_normal", border_mode="same")(x)
#         norm = BatchNormalization(mode=2, axis=3)(conv)
#         return Activation("relu")(norm)

#     return f


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample):
    """Helper to build a BN -> relu -> conv block.

    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(x):
        norm = BatchNormalization(mode=2, axis=3)(x)
        activation = Activation("relu")(norm)
        return Convolution2D(
            nb_filter, nb_row, nb_col, W_regularizer=l2(1e-4), subsample=subsample,
            init="he_normal", border_mode="same")(activation)
    return f


def _bn_relu_deconv(nb_filter, nb_row, nb_col, subsample, output_shape):
    def f(x):
        norm = BatchNormalization(mode=2, axis=3)(x)
        activation = Activation("relu")(norm)
        return Deconvolution2D(
            nb_filter, nb_row, nb_col, W_regularizer=l2(1e-4),
            subsample=subsample, output_shape=output_shape,
            init="he_normal", border_mode="same")(activation)
    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
# def _bottleneck(nb_filters, init_subsample=(1, 1)):
#     def f(x):
#         conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, subsample=init_subsample)(x)
#         conv_3_3 = _bn_relu_conv(nb_filters, 3, 3)(conv_1_1)
#         residual = _bn_relu_conv(nb_filters * 4, 1, 1)(conv_3_3)
#         return _shortcut(x, residual)
#     return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def _basic_block(nb_filters, subsample):
    def f(x):
        residual = _bn_relu_conv(nb_filters, 3, 3, subsample)(x)
        residual = _bn_relu_conv(nb_filters, 3, 3, (1, 1))(residual)
        return _shortcut(x, residual)
    return f


def _deconv_basic_block(nb_filters, subsample, output_shape):
    def f(x):
        residual = _bn_relu_deconv(nb_filters, 3, 3, subsample, output_shape)(x)
        residual = _bn_relu_deconv(nb_filters, 3, 3, (1, 1), output_shape)(residual)
        return _deconv_shortcut(x, residual, output_shape)
    return f


def _shortcut(x, residual):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height).
    # Should be int if network architecture is correctly configured.
    stride_width = x._keras_shape[1] / residual._keras_shape[1]
    stride_height = x._keras_shape[2] / residual._keras_shape[2]
    equal_channels = residual._keras_shape[3] == x._keras_shape[3]

    shortcut = x
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(
            residual._keras_shape[3], 1, 1,
            subsample=(stride_width, stride_height),
            init="he_normal", border_mode="valid")(x)
    return merge([shortcut, residual], mode="sum")


def _deconv_shortcut(x, residual, output_shape):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height).
    # Should be int if network architecture is correctly configured.
    stride_width = residual._keras_shape[1] / x._keras_shape[1]
    stride_height = residual._keras_shape[2] / x._keras_shape[2]
    equal_channels = residual._keras_shape[3] == x._keras_shape[3]

    shortcut = x
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Deconvolution2D(
            residual._keras_shape[3], 1, 1,
            subsample=(stride_width, stride_height),
            output_shape=output_shape,
            init="he_normal", border_mode="valid")(x)
    return merge([shortcut, residual], mode="sum")


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetations, init_subsample):
    def f(x):
        for i in range(repetations):
            subsample = init_subsample if i == 0 else (1, 1)
            x = block_function(nb_filters, subsample)(x)
        return x
    return f


def _deconv_residual_block(block_function, nb_filters, repetations,
                           init_subsample, output_shape):
    def f(x):
        for i in range(repetations):
            subsample = init_subsample if i == 0 else (1, 1)
            x = block_function(nb_filters, subsample, output_shape)(x)
        return x
    return f


def res_autoencoder(input_shape, repetations):
    x = Input(shape=input_shape)

    # ------ encoder -----
    conv1 = Convolution2D(32, 3, 3, init='he_normal', border_mode='same',
                          W_regularizer=l2(1e-4))(x)
    block_func = _basic_block
    block1 = _residual_block(block_func, 32, repetations, (1, 1))(conv1)
    # feature map size (16, 16)
    block2 = _residual_block(block_func, 64, repetations, (2, 2))(block1)
    # feature map size (8, 8)
    block3 = _residual_block(block_func, 64, repetations, (2, 2))(block2)
    # feature map size (8, 8)
    block4 = _residual_block(block_func, 32, repetations, (1, 1))(block3)
    # feature map size (8, 8)
    block5 = _residual_block(block_func, 10, repetations, (1, 1))(block4)

    post_block_norm = BatchNormalization(mode=2, axis=3)(block5)
    encoded = Activation("sigmoid")(post_block_norm)
    print 'encoded shape:', encoded._keras_shape

    encoded = GaussianNoise(0.1)(encoded)

    # ----- decoder -----
    batch_size = tf.shape(encoded)[0]
    h, w, _ = encoded._keras_shape[1:]
    deconv_block_func = _deconv_basic_block
    deconv_block1 = _deconv_residual_block(deconv_block_func, 32, repetations*2,
                                           (1, 1), [batch_size, h, w, 32])(encoded)
    h *=2; w *=2
    deconv_block2 = _deconv_residual_block(deconv_block_func, 64, repetations, (2, 2),
                                           [batch_size, h, w, 64])(deconv_block1)
    h *=2; w *=2
    deconv_block3 = _deconv_residual_block(deconv_block_func, 32, repetations, (2, 2),
                                           [batch_size, h, w, 32])(deconv_block2)

    deconv_block4 = _deconv_residual_block(deconv_block_func, 32, repetations, (1, 1),
                                           [batch_size, h, w, 32])(deconv_block3)
    decoded = BatchNormalization(mode=2, axis=3)(deconv_block4)
    decoded = Activation('relu')(decoded)

    decoded = Convolution2D(3, 3, 3, init='he_normal', border_mode='same',
                            W_regularizer=l2(1e-4), activation='linear')(decoded)
    decoded = BatchNormalization(mode=2, axis=3)(decoded)
    print 'decoded shape:', decoded._keras_shape
    # ----- end -----
    autoencoder = Model(x, decoded)
    return autoencoder


def resnet_cifar10(repetations, input_shape):
    x = Input(shape=input_shape)
    conv1 = Convolution2D(16, 3, 3, init='he_normal', border_mode='same',
                          W_regularizer=l2(1e-4))(x)
    # feature map size (32, 32, 16)

    # Build residual blocks..
    block_fn = _basic_block
    block1 = _residual_block(block_fn, 16, repetations, (1, 1))(conv1)
    # feature map size (16, 16)
    block2 = _residual_block(block_fn, 32, repetations, (2, 2))(block1)
    # feature map size (8, 8)
    block3 = _residual_block(block_fn, 64, repetations, (2, 2))(block2)

    post_block_norm = BatchNormalization(mode=2, axis=3)(block3)
    post_blob_relu = Activation("relu")(post_block_norm)

    # Classifier block
    pool2 = GlobalAveragePooling2D()(post_blob_relu)
    dense = Dense(output_dim=10, init="he_normal",
                  W_regularizer=l2(1e-4), activation="softmax")(pool2)

    model = Model(input=x, output=dense)
    return model


def test_build_model():
    import time
    start = time.time()
    model = resnet_cifar10(repetations=3)
    duration = time.time() - start
    print "{} s to make model".format(duration)

    start = time.time()
    model.output
    duration = time.time() - start
    print "{} s to get output".format(duration)

    start = time.time()
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    duration = time.time() - start
    print "{} s to get compile".format(duration)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, 'res_build.png')
    plot(model, to_file=model_path, show_shapes=True)


if __name__ == '__main__':
    from keras.utils.np_utils import to_categorical
    from keras.datasets import cifar10

    # test_build_model()
    keras.backend.set_session(utils.get_session())

    (train_xs, train_ys), (test_xs, test_ys) = cifar10.load_data()
    train_xs, mean, std = utils.preprocess_cifar10(train_xs)
    test_xs, _, _ = utils.preprocess_cifar10(test_xs)

    train_ys = to_categorical(train_ys)
    test_ys = to_categorical(test_ys)

    model = resnet_cifar10(3, train_xs.shape[1:])
    optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
    model.compile(
        optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False)

    model.fit_generator(datagen.flow(train_xs, train_ys, batch_size=128),
                        samples_per_epoch = train_xs.shape[0],
                        nb_epoch=200,
                        validation_data=(test_xs, test_ys))

    # model.fit(train_xs, train_ys, batch_size=128, nb_epoch=10,
    #           validation_data=(test_xs, test_ys), shuffle=True)
