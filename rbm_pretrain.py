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
    def __init__(self, sess, dataset, rbm, decoder,
                 sampler, sampler_generator, vis_fn):
        self.decoder = decoder
        super(RBMPretrainer, self).__init__(
            sess, dataset, rbm, sampler, sampler_generator, vis_fn)

    def use_pcd(self):
        return hasattr(self.sampler, 'samples')

    def _draw_samples(self):
        burnin = 1000 if self.use_pcd() else 0
        return super(RBMPretrainer, self)._draw_samples(burnin)

    def _save_samples(self, samples, img_path):
        assert hasattr(self, 'decoder'), 'Please set decoder before training'
        samples = self.decoder.predict(samples)
        super(RBMPretrainer, self)._save_samples(samples, img_path)


def pretrain(use_pcd, num_hid, num_epoch, batch_size, lr, cd_k):
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
    num_vis = encoded_dataset.x_shape[0]
    rbm = RBM(num_vis, num_hid, None)

    if use_pcd:
        sample_init_vals = np.random.normal(
        0.0, 1.0, (batch_size,) + encoded_dataset.x_shape)
    else:
        sample_init_vals = encoded_dataset.test_xs[:100]

    def sampler_generator(cd_k=1, init_vals=sample_init_vals):
        # default params to generate samplers for sampling, not training
        return GibbsSampler(init_vals, rbm, cd_k)

    if use_pcd:
        init_vals = np.random.normal(
            0.0, 1.0, (batch_size,) + encoded_dataset.x_shape)
    else:
        init_vals = None

    gibbs_sampler = sampler_generator(cd_k, init_vals)

    trainer = RBMPretrainer(
        sess, encoded_dataset, rbm, ae.decoder,
        gibbs_sampler, sampler_generator, utils.vis_cifar10)

    rbm_folder = 'pretrained_rbm_hid%d_lr%s_batch_%d' % (num_hid, lr, batch_size)
    rbm_folder += '_pcd' if use_pcd else '_cd'
    output_folder = os.path.join(ae_folder, rbm_folder)
    trainer.train(lr, num_epoch, batch_size, output_folder)
    trainer.dump_log(output_folder)


def pretrain_with_cd():
    return pretrain(False, 2000, 100, 100, 0.1, 1)


def pretrain_with_pcd():
    return pretrain(True, 2000, 500, 100, 0.01, 1)

if __name__ == '__main__':
    np.random.seed(666)
    pretrain_with_cd()
    # pretrain_with_pcd()
