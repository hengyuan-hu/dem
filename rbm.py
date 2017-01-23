import numpy as np
import cPickle
import os, sys
import tensorflow as tf


class RBM(object):
    def __init__(self, num_vis, num_hid, name):
        self.num_vis = num_vis
        self.num_hid = num_hid
        self.name = name if name else 'rbm'
        self.vis_shape = [self.num_vis]
        self.hid_shape = [self.num_hid]

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                'weights', shape=[self.num_vis, self.num_hid],
                initializer=tf.random_normal_initializer(0, 0.01))
            self.vbias = tf.get_variable(
                'vbias', shape=[1, self.num_vis],
                initializer=tf.constant_initializer(0.0))
            self.hbias = tf.get_variable(
                'hbias', shape=[1, self.num_hid],
                initializer=tf.constant_initializer(0.0))

    def compute_up(self, vis):
        hid_p = tf.nn.sigmoid(tf.matmul(vis, self.weights) + self.hbias)
        return hid_p

    def compute_down(self, hid):
        vis_p = tf.nn.sigmoid(tf.matmul(hid, tf.transpose(self.weights)) + self.vbias)
        return vis_p

    def sample(self, ps):
        # TODO: change this to static method and others
        rand_uniform = tf.random_uniform(tf.shape(ps), 0, 1)
        samples = tf.to_float(rand_uniform < ps)
        return samples

    def free_energy(self, vis_samples):
        """Compute the free energy defined on visibles.

        return: free energy of shape: [batch_size, 1]
        """
        vbias_term = tf.matmul(vis_samples, self.vbias, transpose_b=True)
        pre_sigmoid_hid_p = tf.matmul(vis_samples, self.weights) + self.hbias
        pre_log_term = 1 + tf.exp(pre_sigmoid_hid_p)
        log_term = tf.log(pre_log_term)
        sum_log = tf.reduce_sum(log_term, reduction_indices=1, keep_dims=True)
        energy = -vbias_term - sum_log
        assert (energy.get_shape().as_list()
                == [vis_samples.get_shape().as_list()[0], 1])
        return energy

    def vhv(self, vis_samples):
        hid_samples = self.sample(self.compute_up(vis_samples))
        vis_p = self.compute_down(hid_samples)
        vis_samples = self.sample(vis_p)
        return vis_p, vis_samples

    def cd(self, vis, k):
        """Contrastive Divergence.

        params: vis is treated as vis_samples.
        """
        def cond(x, vis_p, vis_samples):
            return tf.less(x, k)

        def body(x, vis_p, vis_samples):
            vis_p, vis_samples = self.vhv(vis_samples)
            return x+1, vis_p, vis_samples

        _, vis_p, vis_samples = tf.while_loop(cond, body, [0, vis, vis],
                                              back_prop=False)
        return vis_p, vis_samples

    def get_loss_updates(self, lr, vis, persistent_vis, cd_k):
        # TODO: remove lr from this function and others
        if persistent_vis is not None:
            recon_vis_p, recon_vis_samples = self.cd(persistent_vis, cd_k)
        else:
            recon_vis_p, recon_vis_samples = self.cd(vis, cd_k)

        # use two reduce mean because vis and pst_chain could have different batch_size
        cost = (tf.reduce_mean(self.free_energy(vis))
                - tf.reduce_mean(self.free_energy(recon_vis_samples)))

        loss = self.l2_loss_function(vis)
        return loss, cost, recon_vis_samples

    def l2_loss_function(self, vis):
        recon_vis_p, _ = self.vhv(vis)
        num_dims = len(vis.get_shape().as_list())
        dims = range(num_dims)
        total_loss = tf.reduce_sum(tf.square(vis - recon_vis_p), dims[1:])
        return tf.reduce_mean(total_loss)

    def get_model_parameters(self):
        return {
            'weights': self.weights.eval(),
            'vbias': self.vbias.eval(),
            'hbias': self.hbias.eval()
        }

    def sample_from_rbm(self, num_steps, num_examples, vis):
        def cond(x, vis_p, vis_samples):
            return tf.less(x, num_steps)

        def body(x, vis_p, vis_samples):
            vis_p, vis_samples = self.vhv(vis_samples)
            return x+1, vis_p, vis_samples

        _, prob_imgs, sampled_imgs = tf.while_loop(
            cond, body, [0, vis, vis], back_prop=False)
        return prob_imgs, sampled_imgs


class GaussianRBM(RBM):
    def compute_down(self, hid):
        vis_mean = tf.matmul(hid, tf.transpose(self.weights)) + self.vbias
        return vis_mean

    def sample_gaussian(self, mean):
        dist = tf.contrib.distributions.Normal(mu=mean, sigma=1.0)
        samples = dist.sample((1,))[0]
        return samples

    def free_energy(self, vis_samples):
        """Compute the free energy defined on visibles.

        return: free energy of shape: [batch_size, 1]
        """
        vis_square_sum = 0.5 * tf.reduce_sum(tf.square(vis_samples),
                                             reduction_indices=1, keep_dims=True)
        vbias_term = tf.matmul(vis_samples, self.vbias, transpose_b=True)
        pre_sigmoid_hid_p = tf.matmul(vis_samples, self.weights) + self.hbias
        pre_log_term = 1 + tf.exp(pre_sigmoid_hid_p)
        log_term = tf.log(pre_log_term)
        sum_log = tf.reduce_sum(log_term, reduction_indices=1, keep_dims=True)
        assert  (vbias_term - sum_log).get_shape().as_list() \
            == (vis_samples.get_shape().as_list()[:1] + [1])
        return -vbias_term - sum_log + vis_square_sum

    def vhv(self, vis_samples):
        hid_samples = self.sample(self.compute_up(vis_samples))
        vis_mean = self.compute_down(hid_samples)
        vis_samples = self.sample_gaussian(vis_mean)
        return vis_mean, vis_samples


if __name__ == '__main__':
    import train_rbm

    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print 'usage: python rbm.py pcd/cd cd-k [output_dir]'
        sys.exit()
    else:
        use_pcd = (sys.argv[1] == 'pcd')
        cd_k = int(sys.argv[2])
        output_dir = None if len(sys.argv) == 3 else sys.argv[3]

    (train_xs, _), _, _ = cPickle.load(file('mnist.pkl', 'rb'))
    batch_size = 20
    lr = 0.001 if use_pcd else 0.1
    rbm = RBM(784, 500, output_dir)

    train_rbm.train(rbm, train_xs, lr, 40, batch_size, use_pcd, cd_k, output_dir)
