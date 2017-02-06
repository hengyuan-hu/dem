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


def create_cd_sampler_and_sampler_generator(rbm, cd_k, test_init):
    if cd_k >= 10:
        # when cd-k >= 10, use persistent sampler to generate samples,
        # not reconstructions
        test_init = None

    def sampler_generator():
        """default params to generate samplers for sampling, not training."""
        return GibbsSampler(test_init, rbm, 1)

    sampler = GibbsSampler(None, rbm, cd_k)
    return sampler, sampler_generator


def create_pcd_sampler_and_sampler_generator(rbm, cd_k, chain_shape):
    def sampler_generator(cd_k=1):
        """default params to generate samplers for sampling, not training."""
        init_vals = np.random.normal(0.0, 1.0, chain_shape)
        return GibbsSampler(init_vals, rbm, cd_k)

    sampler = sampler_generator(cd_k)
    return sampler, sampler_generator


def pretrain(sess, rbm, dataset, decoder, train_config, vis_fn, parent_dir):
    if train_config.use_pcd:
        sampler, sampler_generator = create_pcd_sampler_and_sampler_generator(
            rbm, train_config.cd_k, (train_config.batch_size,)+dataset.x_shape)
    else:
        sampler, sampler_generator = create_cd_sampler_and_sampler_generator(
            rbm, train_config.cd_k, dataset.test_xs[:100])

    trainer = RBMPretrainer(sess, dataset, rbm, decoder, sampler,
                            sampler_generator, vis_fn)
    rbm_dir = 'ptrbm_hid%d_%s' % (rbm.num_hid, str(train_config))
    output_dir = os.path.join(parent_dir, rbm_dir)

    train_config.dump_log(output_dir)
    trainer.train(train_config, output_dir)
    trainer.dump_log(output_dir)


def multi_stage_pretrain(sess, rbm, dataset, decoder,
                         train_configs, vis_fn, parent_dir):
    for config in train_cofigs:
        pretrain(sess, rbm, dataset, decoder, config, vis_fn, parent_dir)


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

    train_configs = [
        utils.TrainConfig(lr=0.1, batch_size=100, num_epoch=100, use_pcd=False, cd_k=1),
        utils.TrainConfig(lr=0.05, batch_size=100, num_epoch=100, use_pcd=False, cd_k=3),
        utils.TrainConfig(lr=0.01, batch_size=100, num_epoch=200, use_pcd=False, cd_k=10),
        utils.TrainConfig(lr=0.001, batch_size=100, num_epoch=500, use_pcd=True, cd_k=1),
        utils.TrainConfig(lr=0.001, batch_size=100, num_epoch=500, use_pcd=True, cd_k=5),
    ]
    output_folder = os.path.join(ae_folder, 'multi_stage_pretrain')
    utils.log_train_configs(train_configs, output_folder)

    for config in train_configs:
        pretrain(sess, rbm, encoded_dataset, ae.decoder,
                 config, utils.vis_cifar10, output_folder)
