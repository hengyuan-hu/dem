import numpy as np
import cPickle
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt


def get_session():
    # config = tf.ConfigProto(log_device_placement=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    log_device_placement=True
    return tf.Session(config=config)


class RBM(object):
    def __init__(self, num_vis, num_hid, name):
        self.num_vis = num_vis
        self.num_hid = num_hid
        self.name = name

        with tf.variable_scope(name):
            self.weights = tf.get_variable(
                'weights', shape=[self.num_vis, self.num_hid],
                initializer=tf.random_normal_initializer(0, 0.01))
            self.vbias = tf.get_variable(
                'vbias', shape=[1, self.num_vis],
                initializer=tf.constant_initializer(0.0))
            self.hbias = tf.get_variable(
                'hbias', shape=[1, self.num_hid],
                initializer=tf.constant_initializer(0.0))

        self.params = [self.weights, self.vbias, self.hbias]

    def compute_up(self, vis):
        hid_p = tf.nn.sigmoid(tf.matmul(vis, self.weights) + self.hbias)
        return hid_p

    def compute_down(self, hid):
        vis_p = tf.nn.sigmoid(tf.matmul(hid, tf.transpose(self.weights)) + self.vbias)
        return vis_p
    
    def sample(self, ps):
        rand_uniform = tf.random_uniform(ps.get_shape().as_list(), 0, 1)
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
        assert  (-vbias_term - sum_log).get_shape().as_list() \
            == (vis_samples.get_shape().as_list()[:1] + [1])
        return -vbias_term - sum_log

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
            'weights': self.weights,
            'vbias': self.vbias,
            'hbias': self.hbias
        }

    def sample_from_rbm(self, num_steps, num_examples, vis):
        def cond(x, vis_p, vis_samples):
            return tf.less(x, num_steps)

        def body(x, vis_p, vis_samples):
            vis_p, vis_samples = self.vhv(vis_samples)
            return x+1, vis_p, vis_samples

        _, prob_imgs, sampled_imgs = tf.while_loop(cond, body, [0, vis, vis], back_prop=False)
        
        '''
        with self.sess.as_default():
            _, prob_imgs, sampled_imgs = self.sess.run(
                tf.while_loop(cond, body, [0, vis, vis], back_prop=False),
                feed_dict={num_steps_holder: num_steps, vis: init})
        '''
        return prob_imgs, sampled_imgs


def vis_weights(weights, rows, cols, neuron_shape, output_name=None):
    assert weights.shape[-1] == rows * cols
    f, axarr = plt.subplots(rows, cols)
    for r in range(rows):
        for c in range(cols):
            neuron_idx = r * cols + c
            weight_map = weights[:, neuron_idx].reshape(neuron_shape)
            axarr[r][c].imshow(weight_map, cmap='Greys')
            axarr[r][c].set_axis_off()
    f.subplots_adjust(hspace=0.2, wspace=0.2)
    if output_name is None:
        plt.show()
    else:
        plt.savefig(output_name)

if __name__ == '__main__':
    (train_xs, _), _, _ = cPickle.load(file('mnist.pkl', 'rb'))

    batch_size =  20
    rbm = RBM(784, 500, 'test')
    # rbm.load_model('./rbm.ckpt')
    # rbm.train(train_xs, 0.001, 5, batch_size, True, None, '.')
    # train(self, train_xs, lr, num_epoch, batch_size, use_pcd, cd_k, output_dir):
    # rbm.train(train_xs, 0.1, 40, batch_size, False, 1, None)
    rbm.train(train_xs, 0.001, 40, batch_size, True, 1, None)