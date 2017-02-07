import keras.backend as K
import numpy as np
import os

from dem_trainer import DEMTrainer
from dataset_wrapper import MnistWrapper, Cifar10Wrapper
from rbm import RBM, GibbsSampler
from autoencoder import AutoEncoder
import cifar10_ae
import utils


class RBMPretrainer(DEMTrainer):
    """Encapsulate functions for pretraining RBM in a DEM.

    Mainly used to resolve the difference in sampling.
    """
    def __init__(self, sess, dataset, rbm, decoder, sampler, vis_fn):
        self.decoder = decoder
        super(RBMPretrainer, self).__init__(
            sess, dataset, rbm, sampler, vis_fn)

    def _save_samples(self, samples, img_path):
        assert hasattr(self, 'decoder'), 'Please set decoder before training'
        samples = self.decoder.predict(samples)
        super(RBMPretrainer, self)._save_samples(samples, img_path)


def create_sampler_generator(rbm, init_vals, chain_shape, burnin):
    """create sampler generator to draw sample/reconstruct test."""

    def sampler_generator(init_vals=init_vals):
        if init_vals is None:
            init_vals = np.random.normal(0.0, 1.0, chain_shape)
        return GibbsSampler(init_vals, rbm, 1, burnin)

    return sampler_generator


def pretrain(sess, rbm, dataset, decoder, train_config, vis_fn, parent_dir):
    if train_config.use_pcd:
        chain_shape = (train_config.batch_size,) + dataset.x_shape
        random_init = np.random.normal(0.0, 1.0, chain_shape)
        sampler = GibbsSampler(random_init, rbm, train_config.cd_k, None)
    else:
        sampler = GibbsSampler(None, rbm, train_config.cd_k, None)

    if train_config.draw_samples:
        sampler_generator = create_sampler_generator(
            rbm, None, dataset.test_xs[:100].shape, 1000)
    else:
        sampler_generator = create_sampler_generator(
            rbm, dataset.test_xs[:100], None, 0)

    trainer = RBMPretrainer(sess, dataset, rbm, decoder, sampler, vis_fn)
    rbm_dir = 'ptrbm_hid%d_%s' % (rbm.num_hid, str(train_config))
    output_dir = os.path.join(parent_dir, rbm_dir)

    train_config.dump_log(output_dir)
    trainer.train(train_config, sampler_generator, output_dir)
    trainer.dump_log(output_dir)

    return output_dir

if __name__ == '__main__':
    np.random.seed(666)
    sess = utils.create_session()
    K.set_session(sess)

    ae_folder = 'prod/cifar10_ae2_relu_%d' % cifar10_ae.RELU_MAX
    ae = AutoEncoder(Cifar10Wrapper.load_default(),
                     cifar10_ae.encode, cifar10_ae.decode,
                     cifar10_ae.RELU_MAX, ae_folder)
    ae.build_models(ae_folder) # load model

    encoded_dataset = Cifar10Wrapper.load_from_h5(
        os.path.join(ae_folder, 'encoded_cifar10.h5'))
    assert len(encoded_dataset.x_shape) == 1

    num_hid = 2000
    rbm = RBM(encoded_dataset.x_shape[0], num_hid, None)

    # train_config = utils.TrainConfig(
    #     lr=0.1, batch_size=100, num_epoch=20, use_pcd=False, cd_k=1)
    train_config = utils.TrainConfig(
        lr=0.01, batch_size=100, num_epoch=20, use_pcd=True, cd_k=1)

    output_folder = os.path.join(ae_folder, 'test_pretrain')
    pretrain(sess, rbm, encoded_dataset, ae.decoder,
             train_config, utils.vis_cifar10, output_folder)
