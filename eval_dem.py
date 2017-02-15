from dem import DEM
import cifar10_ae
from dataset_wrapper import Cifar10Wrapper
import utils
import keras.backend as K
import tensorflow as tf
import numpy as np
import os
import keras
from keras.layers import Input, Activation, Dense, Dropout
from keras.models import Model
from keras.utils import np_utils


def save_samples(samples, img_path, vis_fn):
    batch_size = len(samples)
    rows, cols = utils.factorize_number(batch_size)
    vis_fn(samples, rows, cols, img_path)


def run_vhv(sess, dem, test_imgs, n, vis_fn, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    z_samples = dem.encoder.predict(test_imgs)
    z_input = tf.placeholder(tf.float32, z_samples.shape)
    vhv_op = dem.vhv(z_input)

    save_samples(test_imgs, os.path.join(output_folder, '0.png'), vis_fn)

    for i in range(n):
        z_prob, z_samples = sess.run(vhv_op, {z_input: z_samples})

        save_samples(dem.decoder.predict(z_samples),
                     os.path.join(output_folder, '%d.png' % (i+1)),
                     vis_fn)


def compare_free_energy(sess, dem, train_xs, test_xs):
    train_zs = dem.encoder.predict(train_xs)
    test_zs = dem.encoder.predict(test_xs)

    z_input = tf.placeholder(tf.float32, [None] + list(test_zs.shape[1:]))
    fe_op = dem.free_energy(z_input)

    test_fe = sess.run(fe_op, {z_input: test_zs})
    train_fe = sess.run(fe_op, {z_input: train_zs})
    print 'test mean fe:', -test_fe.mean()
    print 'train mean fe:', -train_fe.mean()



def fc1_model(x_shape):
    x = Input(x_shape)
    y = Dense(100, activation='relu')(x)
    y = Dropout(0.8)(y)
    # y = Dense(1024, activation='relu')(x)
    # y = Dropout(0.5)(y)
    y = Dense(10, activation='softmax')(x)
    return Model(x, y)


def semi_supervised(dataset, model):
    opt = keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
    callback_list = []
    # callback_list = [LearningRateScheduler(lr_schedule)]
    num_epoch = 100
    batch_size = 128
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(
        dataset.train_xs, dataset.train_ys,
        nb_epoch=num_epoch,
        batch_size=batch_size,
        validation_data=(dataset.test_xs, dataset.test_ys),
        shuffle=True, callbacks=callback_list)
    # self.history = history.history


if __name__ == '__main__':
    np.random.seed(666)
    sess = utils.create_session()
    K.set_session(sess)

    encoder_weights = 'prod/cifar10_ae3_relu_6/encoder.h5'
    # encoder_weights = 'prod/cifar10_ae3_relu_6/test_up_down_with_cd/epoch_100_encoder.h5'
    # encoder_weights = 'prod/cifar10_ae3_relu_6/test_joint_up_single_down/epoch_400_encoder.h5'
    decoder_weights = encoder_weights.replace('encoder', 'decoder')
    # rbm_weights = encoder_weights.replace('encoder', 'rbm')
    rbm_weights = 'prod/cifar10_ae3_relu_6/ptrbm_scheme1/ptrbm_hid2000_lr0.001_pcd25/epoch_500_rbm.h5'

    dataset = Cifar10Wrapper.load_default()
    dem = DEM.load_from_param_files(
        dataset.x_shape, cifar10_ae.RELU_MAX,
        cifar10_ae.encode, encoder_weights,
        cifar10_ae.decode, decoder_weights,
        rbm_weights)

    utils.initialize_uninitialized_variables_by_keras()

    encoded_dataset = dem.encode(sess, dataset, Cifar10Wrapper)
    encoded_dataset.train_ys = np_utils.to_categorical(encoded_dataset.train_ys)
    encoded_dataset.test_ys = np_utils.to_categorical(encoded_dataset.test_ys)

    # encoded_dataset.dump_to_h5(encoder_weights+'.encoded_dataset.h5')
    regression_model = fc1_model(encoded_dataset.x_shape)
    semi_supervised(encoded_dataset, regression_model)

    # run_vhv(sess, dem, dataset.test_xs[:100], 10, utils.vis_cifar10, 'test_cd')
    # compare_free_energy(sess, dem, dataset.train_xs, dataset.test_xs)
