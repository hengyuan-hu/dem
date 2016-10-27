import numpy as np
import cPickle
import os
import tensorflow as tf
import matplotlib.pyplot as plt


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
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
                'vbias', shape=[self.num_vis],
                initializer=tf.constant_initializer(0.0))
            self.hbias = tf.get_variable(
                'hbias', shape=[self.num_hid],
                initializer=tf.constant_initializer(0.0))
        self.sess = get_session()

    def compute_up(self, vis):
        hid_p = tf.nn.sigmoid(tf.matmul(vis, self.weights) + self.hbias)
        return hid_p

    def compute_down(self, hid):
        vis_p = tf.nn.sigmoid(tf.matmul(hid, tf.transpose(self.weights)) + self.vbias)
        return vis_p

    def sample(self, ps):
        samples = tf.contrib.distributions.Bernoulli(p=ps).sample()
        return tf.to_float(samples)

    def vhv(self, vis_samples):
        hid_samples = self.sample(self.compute_up(vis_samples))
        vis_p = self.compute_down(hid_samples)
        vis_samples = self.sample(vis_p)
        return vis_p, vis_samples
        
    def cd(self, vis, k):
        def cond(x, vis_p, vis_samples):
            return tf.less(x, k)

        def body(x, vis_p, vis_samples):
            vis_p, vis_samples = self.vhv(vis_samples)
            return x+1, vis_p, vis_samples

        _, vis_p, vis_samples = tf.while_loop(cond, body, [0, vis, vis],
                                              back_prop=False)
        return vis_p, vis_samples

    def pcd(self, persistent_hid):
        vis_p = self.compute_down(persistent_hid)
        vis_samples = self.sample(vis_p)
        new_hid_p = self.compute_up(vis_samples)
        pcd_update =  persistent_hid.assign(self.sample(new_hid_p))
        return vis_p, new_hid_p, pcd_update

    def collect_stats(self, vis, hid, batch_size):
        stats_w = tf.matmul(tf.transpose(vis), hid) / batch_size
        stats_vbias = tf.reduce_mean(vis, 0)
        stats_hbias = tf.reduce_mean(hid, 0)
        return stats_w, stats_vbias, stats_hbias

    def compute_gradient(self, vis, batch_size, use_pcd, cd_k, persistent_hid):
        hid_p = self.compute_up(vis)
        pos_stats_w, pos_stats_vbias, pos_stats_hbias = self.collect_stats(
            vis, hid_p, batch_size)

        if use_pcd:
            assert persistent_hid is not None
            recon_vis_p, recon_hid_p, pcd_update = self.pcd(persistent_hid)
        else:
            assert cd_k is not None
            recon_vis_p, recon_vis_samples = self.cd(vis, cd_k)
            # emperically, its better to vis_p for this final step 
            recon_hid_p = self.compute_up(recon_vis_samples)
            pcd_update = None

        neg_stats_w, neg_stats_vbias, neg_stats_hbias = self.collect_stats(
            recon_vis_p, recon_hid_p, batch_size)

        dw = pos_stats_w - neg_stats_w
        dvbias = pos_stats_vbias - neg_stats_vbias
        dhbias = pos_stats_hbias - neg_stats_hbias
        return dw, dvbias, dhbias, pcd_update

    def l2_loss_function(self, vis, batch_size):
        recon_vis_p, _ = self.vhv(vis)
        return tf.reduce_sum(tf.square(vis - recon_vis_p)) / batch_size

    def train_step(self, vis, lr, batch_size, use_pcd, cd_k):
        if use_pcd:
            assert cd_k is None
            persistent_hid = tf.get_variable(
                'persistent_hid', shape=[batch_size, self.num_hid],
                initializer=tf.random_uniform_initializer(0, 1))
        else:
            persistent_hid = None

        dw, dvbias, dhbias, pcd_update = self.compute_gradient(
            vis, batch_size, use_pcd, cd_k, persistent_hid)
        loss = self.l2_loss_function(vis, batch_size)
        # tf.scalar_summary('squared loss', loss)
        if use_pcd:
            assert pcd_update is not None
            updates = tf.group(self.weights.assign_add(lr * dw),
                               self.vbias.assign_add(lr * dvbias),
                               self.hbias.assign_add(lr * dhbias),
                               pcd_update)
        else:
            assert pcd_update is None
            updates = tf.group(self.weights.assign_add(lr * dw),
                               self.vbias.assign_add(lr * dvbias),
                               self.hbias.assign_add(lr * dhbias))
        return loss, updates


    def train(self, train_xs, lr, num_epoch, batch_size, use_pcd, cd_k, output_dir):
        ph_vis = tf.placeholder(tf.float32, (batch_size,) + train_xs.shape[1:], name='vis_input')
        ph_lr = tf.placeholder(tf.float32, (), name='lr')

        # batch_size = tf.Variable(np.float32(batch_size))
        cd_k = None if use_pcd else tf.Variable(cd_k)

        loss, updates = self.train_step(ph_vis, ph_lr, batch_size, use_pcd, cd_k)

        num_batches = len(train_xs) / batch_size
        assert num_batches * batch_size == len(train_xs)
        with self.sess.as_default():
            # merged = tf.merge_all_summaries()
            train_writer = tf.train.SummaryWriter('./train', self.sess.graph)
            tf.initialize_all_variables().run()

            for i in range(num_epoch):
                # np.random.shuffle(train_xs)
                loss_vals = np.zeros(num_batches)
                for b in range(num_batches):
                    batch_xs = train_xs[b * batch_size:(b+1) * batch_size]
                    loss_val, _ = self.sess.run(
                        [loss, updates],
                        feed_dict={ph_vis: batch_xs,
                                   ph_lr: lr})
                    # train_writer.add_summary(summary, i)
                    loss_vals[b] = loss_val
                print 'Train Loss:', loss_vals.mean()

                if output_dir is not None:
                    saver = tf.train.Saver()
                    save_path = saver.save(
                        self.sess,
                        os.path.join(output_dir, '%s-epoch%d.ckpt' % (self.name, i)))
                    print 'Model saved to:', save_path
                    prob_imgs, _ = self.sample_from_rbm(100, 1000)
                    img_path = os.path.join(output_dir, 'epoch%d-plot.png' % i)
                    vis_weights(prob_imgs.T, 10, 10, (28, 28), img_path)
                    params = self.get_model_parameters()
                    params_vis_path = os.path.join(output_dir, 'epoch%d-filters.png' % i)
                    vis_weights(params['weights'][:,:100], 10, 10, (28, 28), params_vis_path)

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
        vis = tf.placeholder(tf.float32, (None, self.num_vis))

        def cond(x, vis_p, vis_samples):
            return tf.less(x, num_steps_holder)

        def body(x, vis_p, vis_samples):
            vis_p, vis_samples = self.vhv(vis_samples)
            return x+1, vis_p, vis_samples

        if init is None:
            init = np.random.normal(0, 1, (num_examples, self.num_vis))

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
    rbm = RBM(784, 500, 'new-cd10')
    # rbm.load_model('./rbm.ckpt')
    # rbm.train(train_xs, 0.001, 5, batch_size, True, None, '.')
    rbm.train(train_xs, 0.1, 40, batch_size, False, 10, 'new-cd10')
