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


class CRBM(object):
    def __init__(self, input_shape, filter_shape, strides, padding, name, params):
        """Initialize a convolutional rbm.

        input_shape: 3D shape of [in_height, in_width, in_channel]
        filter_shape: 4D shape of [filter_height, filter_width, in_channels, out_channels]
        strides: 2D shape of [stride_height, stride_width]
        padding: a string from 'SAME', 'VALID', same as tf.nn.conv2d
        name: defines the name and variable scope of this crbm
        params: a dict of numpy array indicating the initial value of crbm
        """
        assert len(input_shape) == 3
        assert len(filter_shape) == 4
        assert len(strides) == 2
        assert padding == 'SAME' or padding == 'VALID'

        self.vis_shape = list(input_shape)
        self.filter_shape = list(filter_shape)
        self.vbias_shape = list(input_shape[-1:])
        self.hbias_shape = list(filter_shape[-1:])
        self.strides = [1, strides[0], strides[1], 1]
        self.padding = padding
        self.name = name

        weights_init, vbias_init, hbias_init = self._get_initializers(params)
        with tf.variable_scope(name):
            self.weights = tf.get_variable(
                'weights', shape=self.filter_shape, initializer=weights_init)
            self.vbias = tf.get_variable(
                'vbias', shape=self.vbias_shape, initializer=vbias_init)
            self.hbias = tf.get_variable(
                'hbias', shape=self.hbias_shape, initializer=hbias_init)

        self.sess = get_session()

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
        deconv = tf.nn.conv2d_transpose(hid, self.weights, output_shape, self.strides, self.padding)
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
        # assert(self.vbias_shape == [1]), '1 channel input only'
        # vbias_term = tf.reduce_sum(vis_samples, reduction_indices=[1, 2, 3]) * self.vbias
        vbias_term = tf.reduce_sum(vis_samples, reduction_indices=[2,3]) * self.vbias
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

        updates = []
        if persistent_vis is not None:
            updates.append(persistent_vis.assign(recon_vis_samples))

        loss = self.l2_loss_function(vis)
        return loss, cost, updates

    def l2_loss_function(self, vis):
        recon_vis_p, _ = self.vhv(vis)
        num_dims = len(vis.get_shape().as_list())
        dims = range(num_dims)
        total_loss = tf.reduce_sum(tf.square(vis - recon_vis_p), dims[1:])
        return tf.reduce_mean(total_loss)

    def train(self, train_xs, lr, num_epoch, batch_size, use_pcd, cd_k, output_dir):
        train_xs = train_xs.reshape((-1, 28, 28, 1))
        vis_shape = train_xs.shape[1:]    # shape of single image
        batch_shape = (batch_size,) + vis_shape
        num_batches = len(train_xs) / batch_size
        assert num_batches * batch_size == len(train_xs)

        # graph related definitions
        ph_vis = tf.placeholder(tf.float32, batch_shape, name='vis_input')
        ph_lr = tf.placeholder(tf.float32, (), name='lr')
        if use_pcd:
            persistent_vis = tf.get_variable(
                'persistent_vis', shape=batch_shape,
                initializer=tf.random_uniform_initializer(0, 1))
        else:
            persistent_vis = None
            
        loss, cost, updates = self.get_loss_updates(ph_lr, ph_vis, persistent_vis, cd_k)
        opt = tf.train.GradientDescentOptimizer(ph_lr)
        train_step = opt.minimize(cost)

        with self.sess.as_default():
            # merged = tf.merge_all_summaries()
            train_writer = tf.train.SummaryWriter('./train', self.sess.graph)
            tf.initialize_all_variables().run()

            for i in range(num_epoch):
                t = time.time()
                np.random.shuffle(train_xs)
                loss_vals = np.zeros(num_batches)
                for b in range(num_batches):
                    batch_xs = train_xs[b * batch_size:(b+1) * batch_size]

                    loss_vals[b], _, _ = self.sess.run(
                        [loss, train_step, updates], feed_dict={ ph_vis: batch_xs, ph_lr: lr })
                print 'Train Loss:', loss_vals.mean()
                print '... Time took:', time.time() - t
                if output_dir is not None:
                    saver = tf.train.Saver()
                    save_path = saver.save(
                        self.sess,
                        os.path.join(output_dir, '%s-epoch%d.ckpt' % (self.name, i)))
                    print 'Model saved to:', save_path
                    prob_imgs, _ = self.sample_from_rbm(100, 1000)
                    prob_imgs = prob_imgs.reshape(100, -1)
                    img_path = os.path.join(output_dir, 'epoch%d-plot.png' % i)
                    vis_weights(prob_imgs.T, 10, 10, (28, 28), img_path)
                    params = self.get_model_parameters()
                    params_vis_path = os.path.join(output_dir, 'epoch%d-filters.png' % i)
                    vis_weights(
                        params['weights'].reshape((-1, self.filter_shape[-1]))[:,:100],
                        10, 10, (5, 5), params_vis_path)

    def load_model(self, model_path):
        with self.sess.as_default():
            tf.initialize_all_variables().run()
            saver = tf.train.Saver()
            saver.restore(self.sess, model_path)
        print 'Model loaded from:', model_path

    def get_model_parameters(self):
        with self.sess.as_default():
            return {
                'weights': self.weights.eval(),
                'vbias': self.vbias.eval(),
                'hbias': self.hbias.eval()
            }

    def sample_from_rbm(self, num_examples, num_steps, init=None):
        num_steps_holder = tf.placeholder(tf.int32, ())
        vis = tf.placeholder(tf.float32, [num_examples] + self.vis_shape)

        def cond(x, vis_p, vis_samples):
            return tf.less(x, num_steps_holder)

        def body(x, vis_p, vis_samples):
            vis_p, vis_samples = self.vhv(vis_samples)
            return x+1, vis_p, vis_samples

        if init is None:
            init = np.random.normal(0, 1, [num_examples] + self.vis_shape)

        with self.sess.as_default():
            _, prob_imgs, sampled_imgs = self.sess.run(
                tf.while_loop(cond, body, [0, vis, vis], back_prop=False),
                feed_dict={num_steps_holder: num_steps, vis: init})
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
    # crbm(self, input_shape, filter_shape, strides, padding, name, params):
    # params = cPickle.load(open('tf_crbm_init.pkl', 'rb'))
    rbm = CRBM((28, 28, 1), (5, 5, 1, 200), (2, 2), 'VALID', 'crbm-test', {})
    # rbm.train(train_xs, 0.001, 40, batch_size, True, 1, None)
    rbm.train(train_xs, 0.001, 40, batch_size, False, 1, 'crbm-test')
