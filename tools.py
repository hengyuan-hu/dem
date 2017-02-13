import os
import keras
from keras.layers import Input
from keras.models import Model
from keras.utils.visualize_util import plot
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

import utils
import autoencoder
from dataset_wrapper import Cifar10Wrapper
import cifar10_ae


def evalute_encoder(encode_fn, relu_max, weights_file,
                    dataset, dataset_cls):
    encoder = autoencoder.build_model(
        dataset.x_shape, relu_max, encode_fn, None, weights_file)

    encoded_train_xs = encoder.predict(dataset.train_xs)
    encoded_test_xs = encoder.predict(dataset.test_xs)
    encoded_dataset = dataset_cls(encoded_train_xs, dataset.train_ys,
                                  encoded_test_xs, dataset.test_ys)
    plot_path = weights_file + '.encoded_dataset.png'
    encoded_dataset.plot_data_dist(plot_path)


if __name__ == '__main__':
    evalute_encoder(cifar10_ae.encode, 6,
                    'prod/cifar10_new_ae768_relu6/encoder.h5',
                    Cifar10Wrapper.load_default(), Cifar10Wrapper)
