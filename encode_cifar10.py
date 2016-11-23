import cPickle
import keras
from keras.datasets import cifar10
import sys, os
import keras_auto_encoder
import utils


def encode_dataset(dataset, encoder, output_name):
    encoded = encoder.predict(dataset)
    cPickle.dump(encoded, open(output_name, 'wb'))
    print 'dataset encoded and dumped into', output_name


if __name__ == '__main__':
    model_dir = 'noise_02_deep_model1'

    keras.backend.set_session(utils.get_session())
    (train_xs, _), (test_xs, _) = cifar10.load_data()
    train_xs, mean, std = utils.preprocess_cifar10(train_xs)
    test_xs, _, _ = utils.preprocess_cifar10(test_xs)

    encoder, decoder = keras_auto_encoder.load_encoder_decoder(
        train_xs.shape[1:],
        keras_auto_encoder.deep_encoder1, os.path.join(model_dir, 'encoder'),
        keras_auto_encoder.deep_decoder1, os.path.join(model_dir, 'decoder')
    )
    encode_dataset(train_xs, encoder, os.path.join(model_dir, 'encoded_cifar10.pkl'))
