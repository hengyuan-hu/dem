import os
import utils
import keras
from keras.models import Model
from keras.layers import Input, Activation, merge, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

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


def _shortcut(input, residual):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height).
    # Should be int if network architecture is correctly configured.
    stride_width = input._keras_shape[1] / residual._keras_shape[1]
    stride_height = input._keras_shape[2] / residual._keras_shape[2]
    equal_channels = residual._keras_shape[3] == input._keras_shape[3]

    shortcut = input
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(
            nb_filter=residual._keras_shape[3], nb_row=1, nb_col=1,
            subsample=(stride_width, stride_height),
            init="he_normal", border_mode="valid")(input)
    return merge([shortcut, residual], mode="sum")


# # Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetations, init_subsample):
    def f(x):
        for i in range(repetations):
            subsample = init_subsample if i == 0 else (1, 1)
            x = block_function(nb_filters, subsample)(x)
        return x
    return f


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
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    model.fit_generator(datagen.flow(train_xs, train_ys, batch_size=128),
                        samples_per_epoch = train_xs.shape[0],
                        nb_epoch=200,
                        validation_data=(test_xs, test_ys))

    # model.fit(train_xs, train_ys, batch_size=128, nb_epoch=10,
    #           validation_data=(test_xs, test_ys), shuffle=True)
