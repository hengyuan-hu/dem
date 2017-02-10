import os
import keras
from keras.layers import Input
from keras.models import Model
from keras.utils.visualize_util import plot
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

import utils


def _build_model(x_shape, use_noise, relu_max, encode_fn,
                 decode_fn, weights_file):
    assert encode_fn is not None or decode_fn is not None, \
        'At least provide one function to build the model.'
    x = Input(x_shape)
    y = x
    if encode_fn:
        y = encode_fn(x, use_noise, relu_max)
    if decode_fn:
        y = decode_fn(y, use_noise, relu_max)
    model = Model(x, y)
    if weights_file:
        assert os.path.exists(weights_file), '%s does not exist' % weights_file
        model.load_weights(weights_file)
        print 'Model loaded from %s' % weights_file
    return model


_ENCODER_WEIGHTS_FILE = 'encoder.h5'
_DECODER_WEIGHTS_FILE = 'decoder.h5'
_AE_WEIGHTS_FILE = 'ae.h5'


class AutoEncoder(object):
    """Wrapper class for autoencoder."""
    def __init__(self, dataset, encode_fn, decode_fn, relu_max, folder):
        self.dataset = dataset
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.relu_max = relu_max
        self.folder = folder
        self.history = {}
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    @property
    def x_shape(self):
        return self.dataset.x_shape

    @property
    def z_shape(self):
        return self.encoder.get_output_shape_at(-1)[1:]

    def build_models(self, weights_folder=None):
        if weights_folder:
            encoder_weights = os.path.join(weights_folder, _ENCODER_WEIGHTS_FILE)
            decoder_weights = os.path.join(weights_folder, _DECODER_WEIGHTS_FILE)
            ae_weights = os.path.join(weights_folder, _AE_WEIGHTS_FILE)
        else:
            encoder_weights = None
            decoder_weights = None
            ae_weights = None

        self.ae = _build_model(
            self.x_shape, True, self.relu_max,
            self.encode_fn, self.decode_fn, ae_weights)
        self.encoder = _build_model(
            self.x_shape, False, self.relu_max,
            self.encode_fn, None, encoder_weights)
        self.decoder = _build_model(
            self.z_shape, True, self.relu_max,
            None, self.decode_fn, decoder_weights)

    def train(self, batch_size, num_epoch, lr_schedule):
        opt = keras.optimizers.SGD(lr=lr_schedule(0), momentum=0.9, nesterov=True)
        callback_list = [LearningRateScheduler(lr_schedule)]
        self.ae.compile(optimizer=opt, loss='mse')
        history = self.ae.fit(
            self.dataset.train_xs, self.dataset.train_xs,
            nb_epoch=num_epoch,
            batch_size=batch_size,
            validation_data=(self.dataset.test_xs, self.dataset.test_xs),
            shuffle=True, callbacks=callback_list)
        self.history = history.history

    def train_with_data_augmentation(self, batch_size, num_epoch, lr_schedule):
        datagen = ImageDataGenerator(
            width_shift_range=0.125, # randomly shift images horizontally, fraction
            height_shift_range=0.125, # randomly shift images vertically, fraction
            horizontal_flip=True)

        opt = keras.optimizers.SGD(lr=lr_schedule(0), momentum=0.9, nesterov=True)
        callback_list = [LearningRateScheduler(lr_schedule)]
        self.ae.compile(optimizer=opt, loss='mse')
        assert False, 'seems that y is not augmented.'
        # history = self.ae.fit_generator(
        #     datagen.flow(
        #         self.dataset.train_xs,
        #         self.dataset.train_xs,
        #     nb_epoch=num_epoch,
        #     batch_size=batch_size,
        #     validation_data=(self.dataset.test_xs, self.dataset.test_xs),
        #     shuffle=True, callbacks=callback_list)
        self.history = history.history

    def log(self):
        if self.history:
            utils.log_keras_history(
                self.history, os.path.join(self.folder, 'log.txt'))
        else:
            print 'Not trained yet, no training history to log.'

    def save_models(self):
        plot_file = os.path.join(self.folder, 'graph.png')
        plot(self.ae, to_file=plot_file, show_shapes=True)
        print 'model graph is plotted and stored at %s' % plot_file

        ae_weights_file = os.path.join(self.folder, _AE_WEIGHTS_FILE)
        encoder_weights_file = os.path.join(self.folder, _ENCODER_WEIGHTS_FILE)
        decoder_weights_file = os.path.join(self.folder, _DECODER_WEIGHTS_FILE)

        self.ae.save_weights(ae_weights_file)
        print 'ae saved in %s' % ae_weights_file

        ae_weights = self.ae.get_weights()
        encoder_num_layers = len(self.encoder.get_weights())
        decoder_num_layers = len(self.decoder.get_weights())
        assert len(ae_weights) == encoder_num_layers + decoder_num_layers, \
            'Fail to split ae into encoder and decoder, num layer mismatch.'
        self.encoder.set_weights(ae_weights[:encoder_num_layers])
        self.decoder.set_weights(ae_weights[encoder_num_layers:])
        self.encoder.save_weights(encoder_weights_file)
        self.decoder.save_weights(decoder_weights_file)
        print 'encoder and decoder saved'

    def test_models(self, vis_fn):
        rows = 10; cols = 10
        init = self.dataset.test_xs[:rows * cols]
        ae_recon = self.ae.predict(init)
        ed_recon = self.decoder.predict(self.encoder.predict(init))
        vis_fn(init, rows, cols, os.path.join(self.folder, 'test_input.png'))
        vis_fn(ae_recon, rows, cols,
               os.path.join(self.folder, 'test_autoencoder_recon.png'))
        vis_fn(ed_recon, rows, cols,
               os.path.join(self.folder, 'test_encoder_decoder_recon.png'))

    def encode(self, dataset_cls):
        encoded_train_xs = self.encoder.predict(self.dataset.train_xs)
        encoded_test_xs = self.encoder.predict(self.dataset.test_xs)
        print 'in encode: min: %f, max: %f' \
            % (encoded_train_xs.min(), encoded_train_xs.max())
        encoded_dataset = dataset_cls(encoded_train_xs, self.dataset.train_ys,
                                      encoded_test_xs, self.dataset.test_ys)
        return encoded_dataset


if __name__ == '__main__':
    pass
