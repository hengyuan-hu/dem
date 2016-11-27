import vae
import utils
import numpy as np
import keras
import os
from keras.datasets import cifar10


if __name__ == '__main__':
    keras.backend.set_session(utils.get_session())

    num_samples = 100
    input_shape = (1, 1, 1024)

    init = np.random.normal(0, 1, (num_samples,)+input_shape)

    model_name = 'vae2'
    decoder = vae.deep_decoder1(input_shape)
    decoder.load_weights(os.path.join(model_name, 'decoder.h5'))
    print 'decoder loaded'

    pred = decoder.predict(init)
    utils.vis_cifar10(pred, 10, 10, os.path.join(model_name, 'sampled_imgs.png'))

    # test the encode-decode quality
    (train_xs, _), (test_xs, _) = cifar10.load_data()
    train_xs, mean, std = utils.preprocess_cifar10(train_xs)
    test_xs, _, _ = utils.preprocess_cifar10(test_xs)

    auto, loss = vae.deep_model1(train_xs.shape[1:])
    auto.load_weights(os.path.join(model_name, 'encoder_decoder.h5'))

    init = test_xs[:100]
    utils.vis_cifar10(init, 10, 10, os.path.join(model_name, 'auto_input_imgs.png'))
    pred = auto.predict(init)
    utils.vis_cifar10(pred, 10, 10, os.path.join(model_name, 'auto_recon_imgs.png'))
