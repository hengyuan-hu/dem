import keras
import sys, os
import functools

import keras_auto_encoder
import vae
import utils
import keras_utils
import cifar10


def _ae_encode(data, encoder):
    return encoder.predict(data)


def _vae_encode(data, encoder):
    encoded = encoder.predict(data)
    z_mean = encoded[0]
    z_log_var = encoded[1]
    return z_mean


def _encode_dataset(encode_func):
    def f(dataset, encoder, decoder):
        encoded_train_xs = encode_func(dataset.train_xs, encoder)
        encoded_test_xs = encode_func(dataset.test_xs, encoder)
        return cifar10.Cifar10Wrapper(decoder, encoded_train_xs, dataset.train_ys,
                                      encoded_test_xs, dataset.test_ys)
    return f


ae_encode = _encode_dataset(_ae_encode)
vae_encode = _encode_dataset(_vae_encode)


if __name__ == '__main__':
    keras.backend.set_session(utils.get_session())

    model_dir = 'relu_deep_model1_relu_6'
    dataset = cifar10.Cifar10Wrapper()
    # encoder_func = keras_auto_encoder.relu_encoder1
    encoder_func = functools.partial(keras_auto_encoder.relu_encoder1, relu_max=6)

    encoder = keras_utils.load_coder(
        dataset.train_xs.shape[1:], encoder_func,
        os.path.join(model_dir, 'encoder')
    )

    encoded_dataset = ae_encode(dataset, encoder, 'keras_auto_encoder.relu_decoder1')
    encoded_dataset.plot_distribution(os.path.join(model_dir, 'encoded_dist.png'))
    encoded_dataset.dump_to_h5(os.path.join(model_dir, 'encoded_cifar10.h5'))
