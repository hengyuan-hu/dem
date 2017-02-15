import tensorflow as tf
import numpy as np


class GibbsSampler(object):
    """Gibbs sampler for RBM to produce CD / PCD chain."""
    def __init__(self, init_vals, rbm, cd_k, burnin):
        if init_vals is not None:
            self.samples = tf.Variable(init_vals, dtype=tf.float32)
        self.rbm = rbm
        self.cd_k = cd_k
        self.burnin = burnin

    @classmethod
    def create_pcd_sampler(cls, rbm, num_chains, cd_k):
        chain_shape = (num_chains, rbm.num_vis)
        random_init = np.random.normal(0.0, 1.0, chain_shape)
        return cls(random_init, rbm, cd_k, None)

    @classmethod
    def create_cd_sampler(cls, rbm, cd_k):
        return cls(None, rbm, cd_k, None)

    @property
    def is_persistent(self):
        return hasattr(self, 'samples')

    def sample(self, x_data=None):
        if self.is_persistent:
            new_samples = self.samples
        else:
            assert x_data is not None, 'Provide x_data to use CD Gibbs sampler.'
            new_samples = x_data

        for _ in range(self.cd_k):
            vprob, new_samples = self.rbm.vhv(new_samples)
        updates = [self.samples.assign(new_samples)] if self.is_persistent else []
        return vprob, updates


def create_sampler_generator(rbm, init_vals, num_chain, burnin):
    """create sampler generator to draw sample/reconstruct test."""
    if num_chain:
        chain_shape = (num_chain, rbm.num_vis)

    def sampler_generator(init_vals=init_vals):
        if init_vals is None:
            init_vals = np.random.normal(0.0, 1.0, chain_shape)
        return GibbsSampler(init_vals, rbm, 1, burnin)

    return sampler_generator
