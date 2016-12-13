import cPickle
import numpy as np
import keras
from keras.datasets import cifar10
import sys, os
import matplotlib.pyplot as plt

import keras_auto_encoder
import vae
import utils
import keras_utils


def normalize_data(data):
    std = data.std()
    mean = data.mean()
    normed_data = (data - mean) / std
    print 'mean:', mean, ', std:', std
    return normed_data, mean, std


idx2cls = ['airplane', 'automobile', 'bird',
           'cat', 'deer', 'dog', 'frog',
           'horse', 'ship', 'truck']


def split_encode(xs, ys, encoder, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(len(idx2cls)):
        loc = np.where(ys == i)[0]
        cls = xs[loc]
        print cls.shape
        output_name = os.path.join(output_dir, idx2cls[i]+'.pkl')
        encode_dataset(cls, encoder, output_name)


def encode_dataset(dataset, encoder, output_name):
    encoded = encoder.predict(dataset)
    z_mean = encoded[0]
    z_log_var = encoded[1]

    set_size = len(z_mean)
    z_mean = z_mean.reshape((set_size, -1))
    z_log_var = z_log_var.reshape((set_size, -1))
    normed_z_mean, zm_m, zm_std = normalize_data(z_mean)
    normed_z_log_var, zlv_m, zlv_std = normalize_data(z_log_var)

    data = normed_z_mean
    # plt.hist(data)
    # plt.show()
    # data = np.hstack((normed_z_mean, normed_z_log_var))
    print 'dataset shape:', data.shape
    cPickle.dump([data, zm_m, zm_std], open(output_name, 'wb'))
    print 'dataset encoded and dumped into', output_name


if __name__ == '__main__':
    model_dir = 'vae2'

    keras.backend.set_session(utils.get_session())
    (train_xs, train_ys), (test_xs, test_ys) = cifar10.load_data()
    train_xs, _, _ = utils.preprocess_cifar10(train_xs)
    test_xs, _, _ = utils.preprocess_cifar10(test_xs)

    encoder = keras_utils.load_coder(train_xs.shape[1:], vae.deep_encoder1,
                             os.path.join(model_dir, 'encoder'))

    split_encode(train_xs, train_ys, encoder,
                 os.path.join(model_dir, 'split_train'))
    split_encode(test_xs, test_ys, encoder,
                 os.path.join(model_dir, 'split_test'))

    # encode_dataset(train_xs, encoder, os.path.join(model_dir, 'encoded_cifar10.pkl'))
