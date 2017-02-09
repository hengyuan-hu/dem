import tensorflow as tf
import os
import h5py
import utils


class MnistDEM(object):
    def __init__(self, ae, num_hid, params_file):
        """
        params_file: path to the params file to load
        """
        self.ae = ae # TODO: should only use encoder here?
        assert len(self.ae.encoder.get_output_shape_at(-1)) == 2, \
            'Latent (z) space must be 1D.'
        self.num_z = self.ae.encoder.get_output_shape_at(-1)[-1]
        self.num_h = num_hid

        # not necessary to use shared vars here
        if not params_file:
            self.weights = tf.Variable(
                tf.random_normal([self.num_z, self.num_h], 0.0, 0.01))
            self.zbias = tf.Variable(tf.zeros([1, self.num_z]))
            self.hbias = tf.Variable(tf.zeros([1, self.num_h]))
        else:
            with h5py.File(params_file, 'r') as hf:
                self.weights = tf.Variable(np.array(hf.get('weights')))
                self.zbias = tf.Variable(np.array(hf.get('zbias')))
                self.hbias = tf.Variable(np.array(hf.get('hbias')))
        # self.uninited_vars = [self.weights, self.zbias, self.hbias]

    def save_model(self, tf_sess, folder, prefix):
        """Dump model (top level rbm + encoder) into a folder.
        """
        weights, zbias, hbias = tf_sess.run(
            [self.weights, self.zbias, self.hbias])
        rbm_file = os.path.join(folder, prefix+'rbm.h5')
        with h5py.File(rbm_file, 'w') as hf:
            hf.create_dataset('weights', data=weights)
            hf.create_dataset('zbias', data=zbias)
            hf.create_dataset('hbias', data=hbias)
        encoder_file = os.path.join(folder, prefix+'encoder.h5')
        self.ae.encoder.save_weights(encoder_file)

    def free_energy(self, x):
        """Cannot use pred, must rewrite encoder in tensorflow
        and load the pretrained weights from keras model."""
        # this could work
        z = self.ae.encoder(x) # z.shape == (batch_size, num_z)
        return self._free_energy_with_z(z)

    def loss_and_cost(self, x_data, x_model):
        z_data = self.ae.encoder(x_data)
        pos_energy = self._free_energy_with_z(z_data)
        neg_energy = self.free_energy(x_model)
        cost = tf.reduce_mean(pos_energy) - tf.reduce_mean(neg_energy)
        loss = self._z_space_reconstruction_loss(z_data)
        return loss, cost

    def _free_energy_with_z(self, z):
        """Return binary rbm style free energy in shape: [batch_size]"""
        zbias_term = tf.matmul(z, self.zbias, transpose_b=True)
        zbias_term = tf.reshape(zbias_term, [-1]) # flattern
        h_total_input = tf.matmul(z, self.weights) + self.hbias
        softplus_term = utils.softplus(h_total_input)
        sum_softplus = tf.reduce_sum(softplus_term, 1)
        return -zbias_term - sum_softplus

    def _z_space_reconstruction_loss(self, z):
        hprob = tf.nn.sigmoid(tf.matmul(z, self.weights) + self.hbias)
        hsample = utils.sample_bernoulli(hprob)
        zprob = tf.nn.sigmoid(
            tf.matmul(hprob, self.weights, transpose_b=True) + self.zbias)
        assert z.get_shape().ndims == 2, 'z space must be 1D'
        instance_losses = tf.reduce_sum(tf.square(z - zprob), 1)
        return tf.reduce_mean(instance_losses)


if __name__ == '__main__':
    import keras.backend as K
    import numpy as np
    from autoencoder import AutoEncoder
    from dataset_wrapper import MnistWrapper
    import mnist_ae
    import dem_trainer
    import hmc
    import utils

    sess = utils.create_session()
    K.set_session(sess)
    dataset = MnistWrapper.load_default()
    ae = AutoEncoder(dataset, mnist_ae.encode, mnist_ae.decode,
                     mnist_ae.RELU_MAX, 'test/mnist_dem/ae')
    ae.build_models('test/mnist_dem/ae') # load weights

    l1_weights = ae.encoder.layers[1].get_weights()
    print 'l1 weights sum: %s, bias sum: %s' % (
        l1_weights[0].sum(), l1_weights[1].sum())

    train_autoencoder = False
    if train_autoencoder:
        num_epoch = 10
        lr_schedule = utils.generate_decay_lr_schedule(num_epoch, 0.1, 1)
        ae.train(128, num_epoch, lr_schedule)
        ae.save_models()
        ae.test_models(utils.vis_mnist)
        ae.log()

    num_hid = 100
    dem_model = MnistDEM(ae, num_hid, None)

    num_chains = 100
    init_sample = np.random.uniform(0.0, 1.0, (num_chains,) + dataset.x_shape)
    hmc_sampler = hmc.HamiltonianSampler(init_sample,
                                         dem_model.free_energy,
                                         init_stepsize=1.0,
                                         target_accept_rate=0.9,
                                         num_steps=20,
                                         stepsize_min=0.1,
                                         stepsize_max=5.0)

    def sampler_generator():
        init_sample = np.random.uniform(0, 1.0, (num_chains,) + dataset.x_shape)
        return hmc.HamiltonianSampler(init_sample, dem_model.free_energy)

    trainer = dem_trainer.DEMTrainer(
        sess, dataset, dem_model, hmc_sampler, sampler_generator, utils.vis_mnist)
    trainer.train(0.00001, 20, 100, 'test/mnist_dem/dem')
