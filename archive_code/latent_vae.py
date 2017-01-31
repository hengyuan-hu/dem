import keras
from keras.layers import Input, Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.models import Model
from keras.utils.visualize_util import plot
import keras.backend as K

import os
import cPickle
import numpy as np
import tensorflow as tf
import utils
from vae import sampling_gaussian
import keras_utils


def vae1_decoder(input_shape):
    z = Input(shape=input_shape)
    x = Dense(512, activation='relu')(z)
    x = BatchNormalization(mode=2, axis=1)(x)
    x = Dense(1024, activation='linear')(x)
    decoded = BatchNormalization(mode=2, axis=1)(x)
    print 'decoded shape:', decoded._keras_shape
    return Model(z, decoded)


def vae_model1(input_shape):
    input_img = Input(shape=input_shape)
    print 'input shape:', input_img._keras_shape
    x = Dense(1024, activation='relu')(input_img)
    x = BatchNormalization(mode=2, axis=1)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization(mode=2, axis=1)(x)

    latent_dim = 256
    z_mean = Dense(latent_dim, activation='linear')(x)
    z_log_var = Dense(latent_dim, activation='linear')(x)
    z = Lambda(sampling_gaussian, output_shape=(latent_dim,))([z_mean, z_log_var])
    print 'encoded shape:', z._keras_shape

    x = Dense(512, activation='relu')(z)
    x = BatchNormalization(mode=2, axis=1)(x)
    x = Dense(1024, activation='linear')(x)
    decoded = BatchNormalization(mode=2, axis=1)(x)
    print 'decoded shape:', decoded._keras_shape
    vae = Model(input_img, decoded)

    def vae_loss(y, y_pred):
        recon_loss = K.sum(K.square(y_pred - y), axis=[1])
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
                               axis=[1])
        print ('pre average loss shape:',
               recon_loss.get_shape().as_list(),
               kl_loss.get_shape().as_list())
        return K.mean(recon_loss + kl_loss)

    return vae, vae_loss

if __name__ == '__main__':
    keras.backend.set_session(utils.get_session())

    # parent decoder
    decoder_dir = 'vae2'
    train_set = os.path.join(decoder_dir, 'split_train', 'horse.pkl')
    test_set = os.path.join(decoder_dir, 'split_train', 'horse.pkl')
    train_xs, zm_m, zm_std = cPickle.load(file(train_set, 'rb'))
    test_xs, test_zm_m, test_zm_std = cPickle.load(file(test_set, 'rb'))
    print train_xs.shape, train_xs.mean(), train_xs.std()

    # model info
    model_name = 'vae'
    model_path = os.path.join(decoder_dir, model_name)
    auto, loss = vae_model1(train_xs.shape[1:])
    # opt = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
    opt = keras.optimizers.Adam()
    auto.compile(optimizer=opt, loss=loss)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    plot(auto, to_file=os.path.join(model_path, 'graph.png'), show_shapes=True)

    # auto.load_weights(os.path.join(model_path, 'encoder_decoder'))
    # train model
    auto.fit(train_xs, train_xs,
             nb_epoch=250,
             batch_size=128,
             validation_data=(test_xs, test_xs),
             shuffle=True)
    keras_utils.save_model(auto, os.path.join(model_path, 'encoder_decoder'))
    print 'model saved'

    decoder = vae1_decoder((256,))
    weights = auto.get_weights()
    decoder_weights = decoder.get_weights()
    decoder.set_weights(weights[-len(decoder_weights):])
    keras_utils.save_model(decoder, os.path.join(model_path, 'decoder'))
    print 'decoder saved'
