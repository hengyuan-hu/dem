import tensorflow as tf
from autoencoder import build_model
from rbm import RBM


def tf_norm(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x)))


def tf_mean_norm(x):
    return tf.reduce_mean(tf.abs(x))


class DEM(object):
    """Compositional model consists of encoder, assoc memory(rbm), decoder."""
    def __init__(self, ae, rbm, encoder=None, decoder=None):
        # self.ae = ae
        if ae:
            assert False
            self.encoder = ae.encoder
            self.decoder = ae.decoder
        else:
            assert encoder is not None and decoder is not None
            self.encoder = encoder
            self.decoder = decoder
        self.rbm = rbm

        assert len(self.encoder.get_output_shape_at(-1)) == 2, \
            'Latent (z) space must be 1D.'
        assert self.encoder.get_output_shape_at(-1)[-1] == self.num_z

    @classmethod
    def load_from_param_files(cls, x_shape, relu_max,
                              encode_fn, encoder_weights,
                              decode_fn, decoder_weights,
                              rbm_weights):
        with tf.name_scope('encoder'):
            encoder = build_model(
                x_shape, relu_max, encode_fn, None, encoder_weights)
        z_shape = encoder.get_output_shape_at(-1)[1:]
        with tf.name_scope('decoder'):
            decoder = build_model(
                z_shape, relu_max, None, decode_fn, decoder_weights)
        with tf.name_scope('rbm'):
            rbm = RBM(None, None, rbm_weights)
        assert z_shape[0] == rbm.num_vis, ('%s vs %s' % z_shape[0], rbm.num_vis)
        return cls(None, rbm, encoder, decoder)

    @property
    def num_z(self):
        return self.rbm.num_vis

    @property
    def num_h(self):
        return self.rbm.num_hid

    def get_trainable_vars(self, names):
        trainable_vars = []
        for name in names:
            trainable_vars.extend(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name))
        return trainable_vars

    def encode(self, sess, dataset, dataset_cls):
        x = tf.placeholder(tf.float32, [None] + list(dataset.x_shape))
        h = self.rbm._compute_up(self.encoder(x))
        train_set_size = 1000
        # num_batches = len(dataset.train_xs) / batch_size
        encoded_train_xs = sess.run(h, {x: dataset.train_xs[:train_set_size]})
        encoded_test_xs = sess.run(h, {x: dataset.test_xs})
        return dataset_cls(encoded_train_xs, dataset.train_ys[:train_set_size],
                           encoded_test_xs, dataset.test_ys)

    def free_energy(self, z):
        """build the graph to compute free energy given z :: placeholder"""
        return self.rbm.free_energy(z)

    def free_energy_wrt_x(self, x):
        """build the graph to compute free energy given x :: placeholder"""
        z = self.encoder(x)
        fe = tf.reduce_mean(self.rbm.free_energy(z))
        # dfe_dz = tf.gradients(fe, z)[0]
        # grad_norm = tf_mean_norm(dfe_dz)
        # grad_norm = tf.stop_gradient(grad_norm)
        # grad_norm = tf.Print(grad_norm, ['fe grad_norm:', grad_norm])
        return fe# / grad_norm

    def vhv(self, z_samples):
        """z->h->z public interface for GibbsSampler.

        return: z_prob, z_samples
        """
        return self.rbm.vhv(z_samples)

    def rbm_loss_and_cost(self, z_data, z_model):
        """build the graph for assoc memory (rbm) cost and monitoring loss.

        z_data: placeholder holds the encoded training data;
        z_mdoel: placeholder holds the samples from sampler
        """
        return self.rbm.loss_and_cost(z_data, z_model)

    def autoencoder_cost(self, x):
        x_recon = self.decoder(self.encoder(x))
        cost = tf.reduce_mean(tf.square(x_recon - x))
        # dcost_dx_recon = tf.gradients(cost, x_recon)[0]
        # grad_norm = tf_mean_norm(dcost_dx_recon)
        # grad_norm = tf.stop_gradient(grad_norm)
        # grad_norm = tf.Print(grad_norm, ['ae grad_norm:', grad_norm])
        return cost# / grad_norm

    def encoder_cost(self, x, target_z):
        """build the graph for encoder cost.

        x: placeholder for encoder input, should feed decoder-generated images;
        target_z: placeholder for target code, should feed z used by decoder,
                  which are the samples from GibbsSampler;
        """
        z = self.encoder(x)
        cost = tf.reduce_mean(tf.square(target_z - z))
        return cost

    def decoder_cost(self, z, target_x):
        """build the graph for encoder cost.

        z: placeholder for decoder input, should feed encoded training data;
        target_x: placeholder for target output, should feed training data;
        """
        x = self.decoder(z)
        cost = tf.reduce_mean(tf.square(target_x - x))
        return cost

    def save_model(self, sess, output_dir, prefix):
        encoder_weights_file = os.path.join(output_dir, prefix+'encoder.h5')
        decoder_weights_file = os.path.join(output_dir, prefix+'decoder.h5')
        self.encoder.save_weights(encoder_weights_file)
        self.decoder.save_weights(decoder_weights_file)
        self.rbm.save_model(sess, output_dir, prefix)
        print 'model save at', encoder_weights_file


if __name__ == '__main__':
    from dataset_wrapper import Cifar10Wrapper
    from rbm import RBM
    from autoencoder import AutoEncoder
    from dem_trainer import DEMTrainer
    import cifar10_ae
    import gibbs_sampler
    import utils

    import keras.backend as K
    import os
    import h5py
    import numpy as np

    np.random.seed(66699)
    sess = utils.create_session()
    K.set_session(sess)

    dataset = Cifar10Wrapper.load_default()
    ae_folder = 'prod/cifar10_ae3_relu_6/'
    # encoder_weights_file = os.path.join(ae_folder, 'encoder.h5')
    # decoder_weights_file = os.path.join(ae_folder, 'decoder.h5')
    # rbm_params_file = os.path.join(
    #     ae_folder, 'ptrbm_scheme1/ptrbm_hid2000_lr0.001_pcd25/epoch_500_rbm.h5')

    encoder_weights_file = '/home/hhu/Developer/dem/prod/cifar10_ae3_relu_6/test_ae_fe_const_balance/epoch_500_encoder.h5'
    decoder_weights_file = encoder_weights_file.replace('encoder.', 'decoder.')
    rbm_params_file = encoder_weights_file.replace('encoder.', 'rbm.')

    dem = DEM.load_from_param_files(dataset.x_shape, cifar10_ae.RELU_MAX,
                                    cifar10_ae.encode, encoder_weights_file,
                                    cifar10_ae.decode, decoder_weights_file,
                                    rbm_params_file)

    train_config = utils.TrainConfig(
        lr=0.001, batch_size=100, num_epoch=500, use_pcd=True, cd_k=25)

    sampler_generator = gibbs_sampler.create_sampler_generator(dem.rbm, None, 100, 1000)
    sampler = gibbs_sampler.GibbsSampler.create_pcd_sampler(
        dem.rbm, train_config.batch_size, train_config.cd_k)

    output_dir = os.path.join(ae_folder, 'test_ae_fe')
    dem_trainer = DEMTrainer(sess, dataset, dem, utils.vis_cifar10, output_dir)
    dem_trainer.train(train_config, sampler, sampler_generator)
