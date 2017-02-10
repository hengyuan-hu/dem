import tensorflow as tf


class DEM(object):
    """Compositional model consists of encoder, assoc memory(rbm), decoder."""
    def __init__(self, ae, rbm):
        self.ae = ae
        self.rbm = rbm

        assert len(self.ae.encoder.get_output_shape_at(-1)) == 2, \
            'Latent (z) space must be 1D.'
        assert self.ae.encoder.get_output_shape_at(-1)[-1] == self.num_z

    @property
    def num_z(self):
        return self.rbm.num_vis

    @property
    def num_h(self):
        return self.rbm.num_hid

    @property
    def encoder(self):
        return self.ae.encoder

    @property
    def decoder(self):
        return self.ae.decoder

    def free_energy(self, z):
        """build the graph to compute free energy given z :: placeholder"""
        return self.rbm.free_energy(z)

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


# TODO: this is duplicated
def create_sampler_generator(rbm, init_vals, num_chain, burnin):
    """create sampler generator to draw sample/reconstruct test."""
    if num_chain:
        chain_shape = (num_chain, rbm.num_vis)

    def sampler_generator(init_vals=init_vals):
        if init_vals is None:
            init_vals = np.random.normal(0.0, 1.0, chain_shape)
        return GibbsSampler(init_vals, rbm, 1, burnin)

    return sampler_generator


if __name__ == '__main__':
    from dataset_wrapper import Cifar10Wrapper
    from rbm import RBM, GibbsSampler
    from autoencoder import AutoEncoder
    from real_dem_trainer import DEMTrainer
    import cifar10_ae
    import utils

    import keras.backend as K
    import os
    import h5py
    import numpy as np

    np.random.seed(666)
    sess = utils.create_session()
    K.set_session(sess)

    dataset = Cifar10Wrapper.load_default()
    ae_folder = 'prod/cifar10_ae3_relu_6/'
    ae = AutoEncoder(dataset, cifar10_ae.encode, cifar10_ae.decode,
                     cifar10_ae.RELU_MAX, ae_folder)
    ae.build_models(ae_folder)

    rbm_params_file = os.path.join(
        ae_folder, 'ptrbm_scheme1/ptrbm_hid2000_lr0.001_pcd25/epoch_500_rbm.h5')
    rbm = RBM(None, None, rbm_params_file)
    dem = DEM(ae, rbm)

    train_config = utils.TrainConfig(
        lr=0.001, batch_size=100, num_epoch=500, use_pcd=True, cd_k=25)
    sampler_generator = create_sampler_generator(rbm, None, 100, 1000)
    # pcd sampler
    chain_shape = (train_config.batch_size, rbm.num_vis)
    random_init = np.random.normal(0.0, 1.0, chain_shape)
    sampler = GibbsSampler(random_init, rbm, train_config.cd_k, None)

    output_dir = os.path.join(ae_folder, 'test_up_down')
    dem_trainer = DEMTrainer(sess, dataset, dem, utils.vis_cifar10, output_dir)
    # dem_trainer.test_decode()
    # dem_trainer._test_init()
    dem_trainer.train(train_config, sampler, sampler_generator)
