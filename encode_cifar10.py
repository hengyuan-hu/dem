import cPickle
import keras
from keras.datasets import cifar10

import keras_auto_encoder
import utils


def encode_dataset(dataset, encoder, output_name):
    encoded = encoder.predict(dataset)
    cPickle.dump(encoded, open(output_name, 'wb'))
    print 'dataset encoded and dumped into', output_name


if __name__ == '__main__':
    keras.backend.set_session(utils.get_session())

    (train_xs, _), (test_xs, _) = cifar10.load_data()
    train_xs, mean, std = utils.preprocess_cifar10(train_xs)
    test_xs, _, _ = utils.preprocess_cifar10(test_xs)

    encoder, _ = keras_auto_encoder.load_encoder_decoder(
        train_xs.shape[1:], keras_auto_encoder.deep_encoder1, 'noise_deep_encoder1',
        keras_auto_encoder.deep_decoder1, 'noise_deep_decoder1')

    encode_dataset(train_xs, encoder, 'noise_deep_encoder1_encoded_cifar10.pkl')
