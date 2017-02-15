import os
import h5py
import numpy as np
import tensorflow as tf
import utils


class RBM(object):
    def __init__(self, num_vis, num_hid, params_file):
        if not params_file:
            assert num_vis and num_hid
            self.num_vis = num_vis
            self.num_hid = num_hid
            self.weights = tf.Variable(
                tf.random_normal([self.num_vis, self.num_hid], 0.0, 0.01), name='w')
            self.vbias = tf.Variable(tf.zeros([1, self.num_vis]), name='vbias')
            self.hbias = tf.Variable(tf.zeros([1, self.num_hid]), name='hbias')
        else:
            with h5py.File(params_file, 'r') as hf:
                self.weights = tf.Variable(np.array(hf.get('weights')), name='w')
                self.vbias = tf.Variable(np.array(hf.get('vbias')), name='vbias')
                self.hbias = tf.Variable(np.array(hf.get('hbias')), name='hbias')
            self.num_vis, self.num_hid = self.weights.get_shape().as_list()

    def save_model(self, tf_sess, folder, prefix):
        weights, vbias, hbias = tf_sess.run(
            [self.weights, self.vbias, self.hbias])
        rbm_file = os.path.join(folder, prefix+'rbm.h5')
        with h5py.File(rbm_file, 'w') as hf:
            hf.create_dataset('weights', data=weights)
            hf.create_dataset('vbias', data=vbias)
            hf.create_dataset('hbias', data=hbias)

    def free_energy(self, vis_samples):
        """Compute the free energy defined on visibles.

        return: free energy of shape: [batch_size, 1]
        """
        vbias_term = tf.matmul(vis_samples, self.vbias, transpose_b=True)
        vbias_term = tf.reshape(vbias_term, [-1]) # flattern
        h_total_input = tf.matmul(vis_samples, self.weights) + self.hbias
        softplus_term = utils.softplus(h_total_input)
        sum_softplus = tf.reduce_sum(softplus_term, 1)
        return -vbias_term - sum_softplus

    def vhv(self, vis_samples):
        hid_samples = utils.sample_bernoulli(self._compute_up(vis_samples))
        vprob = self._compute_down(hid_samples)
        vis_samples = utils.sample_bernoulli(vprob)
        return vprob, vis_samples

    def loss_and_cost(self, vis_data, vis_model):
        cost = (tf.reduce_mean(self.free_energy(vis_data))
                - tf.reduce_mean(self.free_energy(vis_model)))
        loss = self._l2_loss_function(vis_data)
        return loss, cost

    def _compute_up(self, vis):
        hprob = tf.nn.sigmoid(tf.matmul(vis, self.weights) + self.hbias)
        return hprob

    def _compute_down(self, hid):
        vprob = tf.nn.sigmoid(
            tf.matmul(hid, self.weights, transpose_b=True) + self.vbias)
        return vprob

    def _l2_loss_function(self, vis):
        recon_vprob, _ = self.vhv(vis)
        num_dims = vis.get_shape().ndims
        dims = range(num_dims)
        instance_loss = tf.reduce_sum(tf.square(vis - recon_vprob), dims[1:])
        return tf.reduce_mean(instance_loss)


# class GaussianRBM(RBM):
#     def _compute_down(self, hid):
#         vis_mean = tf.matmul(hid, tf.transpose(self.weights)) + self.vbias
#         return vis_mean

#     def sample_gaussian(self, mean):
#         dist = tf.contrib.distributions.Normal(mu=mean, sigma=1.0)
#         samples = dist.sample((1,))[0]
#         return samples

#     def free_energy(self, vis_samples):
#         """Compute the free energy defined on visibles.

#         return: free energy of shape: [batch_size, 1]
#         """
#         vis_square_sum = 0.5 * tf.reduce_sum(tf.square(vis_samples),
#                                              reduction_indices=1, keep_dims=True)
#         vbias_term = tf.matmul(vis_samples, self.vbias, transpose_b=True)
#         h_total_input = tf.matmul(vis_samples, self.weights) + self.hbias
#         pre_log_term = 1 + tf.exp(h_total_input)
#         log_term = tf.log(pre_log_term)
#         sum_log = tf.reduce_sum(log_term, reduction_indices=1, keep_dims=True)
#         assert  (vbias_term - sum_log).get_shape().as_list() \
#             == (vis_samples.get_shape().as_list()[:1] + [1])
#         return -vbias_term - sum_log + vis_square_sum

#     def vhv(self, vis_samples):
#         hid_samples = utils.sample_bernoulli(self._compute_up(vis_samples))
#         vis_mean = self._compute_down(hid_samples)
#         vis_samples = self.sample_gaussian(vis_mean)
#         return vis_mean, vis_samples
