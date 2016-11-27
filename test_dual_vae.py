import keras
from keras.datasets import cifar10

import numpy as np
import os
import cPickle
import vae
import latent_vae
import utils


if __name__ == '__main__':
    keras.backend.set_session(utils.get_session())

    # deocder
    decoder_name = 'vae2'
    decoder_input_shape = (1, 1, 1024)
    decoder = vae.deep_decoder1(decoder_input_shape)
    decoder.load_weights(os.path.join(decoder_name, 'decoder.h5'))
    print 'decoder loaded'

    train_set = os.path.join(decoder_name, 'split_train', 'horse.pkl')
    train_xs, zm_m, zm_std = cPickle.load(file(train_set, 'rb'))

    # latent decoder
    latent_decoder_name = 'vae'
    latent_decoder_path = os.path.join(decoder_name, latent_decoder_name)
    ld_input_shape = (256,)
    latent_decoder = latent_vae.vae1_decoder(ld_input_shape)
    latent_decoder.load_weights(os.path.join(latent_decoder_path, 'decoder.h5'))

    # sampling
    num_samples = 100
    init = np.random.normal(0, 1, (num_samples,) + ld_input_shape)
    latent = latent_decoder.predict(init)
    # print zm_std
    latent = latent * zm_std + zm_m
    latent = latent.reshape((-1,) + decoder_input_shape)
    pred = decoder.predict(latent)
    utils.vis_cifar10(pred, 10, 10, os.path.join(latent_decoder_path, 'sampled_imgs.png'))
