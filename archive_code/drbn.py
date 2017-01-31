import numpy as np
import cPickle
import os, sys
import tensorflow as tf
import matplotlib.pyplot as plt
from rbm import RBM, GaussianRBM
from crbm import CRBM, GaussianCRBM
import train_rbm


class DRBN(object):
    def __init__(self, vis_shape, name='drbn'):
        self.vis_shape = list(vis_shape)
        self.last_conv = None    # index of the final crbm layer
        self.num_rbm = 0
        self.rbm_list = []
        self.name = name

    def add_conv_layer(self, filter_shape, stride, padding, name,
                       use_gaussian=False, params={}):
        assert self.last_conv is None, 'cannot add conv layer after fc layers'

        if self.num_rbm == 0:
            vis_shape = self.vis_shape
        else:
            vis_shape = self.rbm_list[-1].hid_shape

        if use_gaussian:
            rbm = GaussianCRBM(vis_shape, filter_shape, stride, padding, name, params)
        else:
            rbm = CRBM(vis_shape, filter_shape, stride, padding, name, params)
        self.num_rbm += 1
        self.rbm_list.append(rbm)

    def add_fc_layer(self, num_hid, name, use_gaussian=False):
        if self.last_conv is None:
            self.last_conv = self.num_rbm - 1

        if self.num_rbm == 0:
            num_vis = int(np.prod(self.vis_shape))
        else:
            num_vis = int(np.prod(self.rbm_list[-1].hid_shape))

        if use_gaussian:
            rbm = GaussianRBM(num_vis, num_hid, name)
        else:
            rbm = RBM(num_vis, num_hid, name)
        self.num_rbm += 1
        self.rbm_list.append(rbm)

    def print_network(self):
        print '-----Network Dimensions-----'
        for i, rbm in enumerate(self.rbm_list):
            print rbm.vis_shape
            if i == self.last_conv:
                print rbm.hid_shape
        print self.rbm_list[-1].hid_shape
        print '----------------------------'

    def compute_up(self, vis_samples):
        """Returns last hidden sample and samples list."""
        samples_list = [vis_samples]
        for i in range(self.num_rbm):
            rbm = self.rbm_list[i]
            vis_samples = rbm.sample(rbm.compute_up(vis_samples))
            if (i + 1 < self.num_rbm and
                vis_samples.get_shape().as_list()[1:] != self.rbm_list[i+1].vis_shape):
                assert i == self.last_conv
                vis_samples = tf.reshape(vis_samples, [-1] + self.rbm_list[i+1].vis_shape)
            vis_samples = tf.stop_gradient(vis_samples)
            samples_list.append(vis_samples)
        return samples_list

    def compute_down(self, hid_samples):
        samples_list = [hid_samples]
        k = self.num_rbm
        for i in range(self.num_rbm-1, -1, -1):
            rbm = self.rbm_list[i]
            if hid_samples.get_shape().as_list()[1:] != rbm.hid_shape:
                assert i == self.last_conv
                hid_samples = tf.reshape(hid_samples, [-1] + rbm.hid_shape)
            if type(rbm) == GaussianRBM or type(rbm) == GaussianCRBM:
                hid_samples = rbm.sample_gaussian(rbm.compute_down(hid_samples))
            else:
                hid_samples = rbm.sample(rbm.compute_down(hid_samples))
            hid_samples = tf.stop_gradient(hid_samples)
            samples_list.append(hid_samples)
        return samples_list[::-1]

    def vhv(self, vis_samples):
        vis_samples_list = self.compute_up(vis_samples)
        recon_samples_list = self.compute_down(vis_samples_list[-1])
        assert len(vis_samples_list) == len(recon_samples_list)
        return recon_samples_list

    def cd(self, vis, k):
        """Contrastive Divergence.

        params: vis is treated as vis_samples.
        return: samples_list of the first compute up and the last compute_down
        """
        vis_samples_list = self.compute_up(vis)
        recon_samples_list = self.compute_down(vis_samples_list[-1])

        for _ in range(k-1):
            recon_samples_list = self.vhv(recon_samples_list[0])
        return vis_samples_list, recon_samples_list

    def get_loss_updates(self, lr, vis, persistent_vis, cd_k):
        if persistent_vis is not None:
            vis_samples_list = self.compute_up(vis)
            _, recon_samples_list = self.cd(persistent_vis, cd_k)
        else:
            vis_samples_list, recon_samples_list = self.cd(vis, cd_k)

        energy_vis = tf.zeros([], tf.float32)
        energy_recon = tf.zeros([], tf.float32)

        assert len(recon_samples_list) == self.num_rbm + 1
        for i in range(self.num_rbm):
            # no need to compare free energy of the last layer
            vis_samples = vis_samples_list[i]
            recon_samples = recon_samples_list[i]
            rbm = self.rbm_list[i]
            energy_vis += tf.reduce_mean(rbm.free_energy(vis_samples))
            energy_recon += tf.reduce_mean(rbm.free_energy(recon_samples))

        cost = energy_vis - energy_recon

        loss = self.l2_loss(vis)
        return loss, cost, recon_samples_list[0]

    def _one_rbm_compute_down(self, rbm, samples):
        """One step compute down, return prob, no samples.

        Should ONLY be used for generating prob imgs for loss and plot purpose!
        """
        if samples.get_shape().as_list()[1:] != rbm.hid_shape:
            samples = tf.reshape(samples, [-1] + rbm.hid_shape)
        return rbm.compute_down(samples)

    def l2_loss(self, vis):
        recon_samples_list = self.vhv(vis)
        recon_vis_p = self._one_rbm_compute_down(self.rbm_list[0], recon_samples_list[1])
        num_dims = len(vis.get_shape().as_list())
        dims = range(num_dims)
        total_loss = tf.reduce_sum(tf.square(vis - recon_vis_p), dims[1:])
        return tf.reduce_mean(total_loss)

    def sample_from_rbm(self, num_steps, vis):
        def cond(x, vis_p, vis_samples):
            return tf.less(x, num_steps)

        def body(x, vis_p, vis_samples):
            recon_samples_list = self.vhv(vis_samples)
            vis_p = self._one_rbm_compute_down(self.rbm_list[0], recon_samples_list[1])
            vis_samples = recon_samples_list[0]
            return x+1, vis_p, vis_samples

        _, prob_imgs, sampled_imgs = tf.while_loop(cond, body, [0, vis, vis], back_prop=False)
        return prob_imgs, sampled_imgs

if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print 'usage: python drbn.py pcd/cd cd-k [output_dir]'
        sys.exit()
    else:
        use_pcd = (sys.argv[1] == 'pcd')
        cd_k = int(sys.argv[2])
        output_dir = None if len(sys.argv) == 3 else sys.argv[3]

    (train_xs, _), _, _ = cPickle.load(file('mnist.pkl', 'rb'))
    train_xs = train_xs.reshape((-1, 28, 28, 1))
    batch_size = 20
    lr = 0.001 if use_pcd else 0.1

    # drbn = DRBN([784])
    drbn = DRBN([28, 28, 1])
    drbn.add_conv_layer((5, 5, 1, 64), (2, 2), 'SAME', 'conv1')
    drbn.add_conv_layer((5, 5, 64, 64), (2, 2), 'SAME', 'conv2')
    drbn.add_fc_layer(500, 'fc1')
    drbn.print_network()

    train_rbm.train(drbn, train_xs, lr, 40, batch_size, use_pcd, cd_k, output_dir)
