import numpy as np
import cPickle
import os, sys
import tensorflow as tf
import utils


class CRBM(object):
    def __init__(self, vis_shape, filter_shape, strides, padding, name, params):
        """Initialize a convolutional rbm.

        vis_shape: 3D shape of [in_height, in_width, in_channel]
        filter_shape: 4D shape of [filter_height, filter_width, in_channels, out_channels]
        strides: 2D shape of [stride_height, stride_width]
        padding: a string from 'SAME', 'VALID', same as tf.nn.conv2d
        name: defines the name and variable scope of this crbm
        params: a dict of numpy array indicating the initial value of crbm
        """
        assert len(vis_shape) == 3
        assert len(filter_shape) == 4
        assert len(strides) == 2
        assert padding == 'SAME' or padding == 'VALID'

        self.vis_shape = list(vis_shape)
        self.filter_shape = list(filter_shape)
        self.vbias_shape = list(vis_shape[-1:])
        self.hbias_shape = list(filter_shape[-1:])
        self.strides = [1, strides[0], strides[1], 1]
        self.padding = padding
        self.name = name if name else 'crbm'
        self.hid_shape = utils.conv_output_shape(
            self.vis_shape, self.filter_shape, strides, self.padding)

        weights_init, vbias_init, hbias_init = self._get_initializers(params)
        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                'weights', shape=self.filter_shape, initializer=weights_init)
            self.vbias = tf.get_variable(
                'vbias', shape=self.vbias_shape, initializer=vbias_init)
            self.hbias = tf.get_variable(
                'hbias', shape=self.hbias_shape, initializer=hbias_init)

    def _get_initializers(self, params):
        if 'weights' in params:
            weights_init_val = params['weights']
            print list(weights_init_val.shape), 'vs', self.filter_shape
            assert list(weights_init_val.shape) == self.filter_shape
            weights_initializer = tf.constant_initializer(weights_init_val)
        else:
            weights_initializer = tf.random_normal_initializer(0, 0.01)

        if 'vbias' in params:
            vbias_init_val = params['vbias']
            assert list(vbias_init_val.shape) == self.vbias_shape
            vbias_initializer = tf.constant_initializer(vbias_init_val)
        else:
            vbias_initializer = tf.constant_initializer(0.0)

        if 'hbias' in params:
            hbias_init_val = params['hbias']
            assert list(hbias_init_val.shape) == self.hbias_shape
            hbias_initializer = tf.constant_initializer(hbias_init_val)
        else:
            hbias_initializer = tf.constant_initializer(0.0)

        return weights_initializer, vbias_initializer, hbias_initializer

    def compute_up(self, vis):
        conv = tf.nn.conv2d(vis, self.weights, self.strides, self.padding)
        conv = tf.nn.bias_add(conv, self.hbias)
        print 'conv_shape:', conv.get_shape().as_list()
        hid_p = tf.nn.sigmoid(conv)
        return hid_p

    def compute_down(self, hid):
        output_shape = hid.get_shape().as_list()[:1] + self.vis_shape
        deconv = tf.nn.conv2d_transpose(hid, self.weights, output_shape,
                                        self.strides, self.padding)
        deconv = tf.nn.bias_add(deconv, self.vbias)
        print 'deconv_shape:',  deconv.get_shape().as_list()
        print 'original vis_shape', self.vis_shape
        vis_p = tf.nn.sigmoid(deconv)
        return vis_p

    def sample(self, ps):
        rand_uniform = tf.random_uniform(ps.get_shape().as_list(), 0, 1)
        samples = tf.to_float(rand_uniform < ps)
        return samples

    def free_energy(self, vis_samples):
        """Compute the free energy defined on visibles.

        return: free energy of shape: [batch_size, 1]
        """
        assert(len(vis_samples.get_shape().as_list()) == 4)
        vbias_term = tf.reduce_sum(vis_samples, reduction_indices=[1, 2]) * self.vbias
        vbias_term = tf.reduce_sum(vbias_term, reduction_indices=[1])

        conv = tf.nn.conv2d(vis_samples, self.weights, self.strides, self.padding)
        pre_sigmoid_hid_p = tf.nn.bias_add(conv, self.hbias)
        pre_log_term = 1 + tf.exp(pre_sigmoid_hid_p)
        log_term = tf.log(pre_log_term)
        sum_log = tf.reduce_sum(log_term, reduction_indices=[1,2,3])
        assert sum_log.get_shape().as_list() == vbias_term.get_shape().as_list()
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

        # treat recon_vis_samples as constant during gradient comp
        recon_vis_samples = tf.stop_gradient(recon_vis_samples)

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

    def sample_from_rbm(self, num_steps, num_exmaples, vis):
        def cond(x, vis_p, vis_samples):
            return tf.less(x, num_steps)

        def body(x, vis_p, vis_samples):
            vis_p, vis_samples = self.vhv(vis_samples)
            return x+1, vis_p, vis_samples

        _, prob_imgs, sampled_imgs = tf.while_loop(cond, body, [0, vis, vis], back_prop=False)
        return prob_imgs, sampled_imgs


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
    train_xs = train_xs.reshape((-1, 28, 28, 1))
    batch_size = 20
    lr = 0.001 # if use_pcd else 0.1
    rbm = CRBM((28, 28, 1), (5, 5, 1, 200), (2, 2), 'SAME', output_dir, {})
    train_rbm.train(rbm, train_xs, lr, 40, batch_size, use_pcd, cd_k, output_dir)
