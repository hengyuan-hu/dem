import numpy as np
import cPickle
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import utils
import keras_auto_encoder


def _copy_tensor_list(tensors):
    new_tensors = []
    for t in tensors:
        new_tensors.append(tf.zeros_like(t) + t)
    return new_tensors


class DBM(object):
    def __init__(self, num_units, name=None):
        self.num_units = num_units
        self.name = name if name is not None else 'dbm'
        self.weights = []
        self.biases = []

        with tf.variable_scope(self.name):
            for i in range(self.num_layers):
                if i < self.num_layers - 1:
                    self.weights.append(tf.get_variable(
                        'weights_%d' % i, shape=self.num_units[i:i+2],
                        initializer=tf.random_normal_initializer(0, 0.01)))
                self.biases.append(tf.get_variable(
                    'bias_%d' % i, shape=[self.num_units[i]],
                    initializer=tf.constant_initializer(0.0)))

    @property
    def num_layers(self):
        return len(self.num_units)

    def init_states(self, vis):
        """Return a list of states filled with tensors (not Variables)."""
        batch_size = tf.shape(vis)[0]
        states = [vis]
        for i in range(1, self.num_layers):
            states.append(tf.random_uniform([batch_size, self.num_units[i]], 0, 1))
        return states

    def _compute_probs(self, states, offset):
        """Compute prob (in place) given a list of tensors."""
        for i in range(offset, self.num_layers, 2):
            pre_sigmoid = tf.zeros_like(states[i]) + self.biases[i]
            if i > 0:
                pre_sigmoid += tf.matmul(states[i-1], self.weights[i-1])
            if i+i < self.num_layers:
                pre_sigmoid += tf.matmul(states[i+1], tf.transpose(self.weights[i]))
            states[i] = tf.nn.sigmoid(pre_sigmoid)

    def _sample_probs(self, states, offset):
        """In place bernoulli sampling given a list of probs (in tensors)."""
        for i in range(offset, self.num_layers, 2):
            states[i] = utils.sample_bernoulli(states[i])

    def mean_field(self, k, states):
        """In place mean-field updates till convergence."""
        for i in range(k):
            self._compute_probs(states, 1)
            self._compute_probs(states, 2)

    def pcd(self, k, states):
        """In place pcd updates, input should be a list of tensor."""
        for i in range(k):
            self._compute_probs(states, 1)
            self._sample_probs(states, 1)
            self._compute_probs(states, 0)
            self._sample_probs(states, 0)

    def collect_stats(self, states):
        """Collect stats given mean filed/pcd states; params not changed."""
        num_chains = tf.to_float(tf.shape(states[0])[0])
        stats_ws = []
        stats_biases = []

        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                layer_stats = tf.matmul(tf.transpose(states[i]), states[i+1]) / num_chains
                stats_ws.append(layer_stats)
            stats_biases.append(tf.reduce_mean(states[i], reduction_indices=[0]))
        return stats_ws, stats_biases

    def compute_gradient(self, mf_states, pcd_states, mf_k, pcd_k):
        """Compute gradients, mf_states and pcd_states will be updated in place."""
        self.mean_field(mf_k, mf_states)
        self.pcd(pcd_k, pcd_states)
        pos_stats_ws, pos_stats_biases = self.collect_stats(mf_states)
        neg_stats_ws, neg_stats_biases = self.collect_stats(pcd_states)

        dws = []
        dbiases = []
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                dws.append(pos_stats_ws[i] - neg_stats_ws[i])
            dbiases.append(pos_stats_biases[i] - neg_stats_biases[i])
        return dws, dbiases

    def compute_loss(self, vis):
        states = self.init_states(vis)
        self._compute_probs(states, 1)
        self._sample_probs(states, 1)
        self._compute_probs(states, 0)
        recon_vprob = states[0]
        loss = vis * tf.log(recon_vprob) + (1-vis) * tf.log(1 - recon_vprob)
        return - tf.reduce_mean(tf.reduce_sum(loss, reduction_indices=[1]))

    def train_step(self, lr, vis, pcd_states, mf_k, pcd_k):
        mf_states = self.init_states(vis)
        new_pcd_states = _copy_tensor_list(pcd_states)

        dws, dbiases  = self.compute_gradient(mf_states, new_pcd_states, mf_k, pcd_k)
        loss = self.compute_loss(vis)

        updates = []
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                updates.append(self.weights[i].assign_add(lr * dws[i]))
            updates.append(self.biases[i].assign_add(lr * dbiases[i]))
        return loss, updates, new_pcd_states

    def sample_from_dbm(self, num_examples, num_steps):
        init = tf.random_uniform([num_examples, self.num_units[0]], 0, 1)
        states = self.init_states(init)

        self.pcd(num_steps-1, states)
        self._compute_probs(states, 1)
        self._sample_probs(states, 1)
        self._compute_probs(states, 0)
        return states[0]


def train(dbm, xs, init_lr, num_epoch, batch_size,
          mf_k, pcd_k, pcd_chain_size, output_dir):
    num_batches = len(xs) / batch_size
    assert num_batches * batch_size == len(xs)

    vis = tf.placeholder(tf.float32, (None,) + xs.shape[1:], name='vis_input')
    lr = tf.placeholder(tf.float32, (), name='lr')

    pcd_states = [tf.placeholder(tf.float32, (pcd_chain_size, i)) for i in dbm.num_units]
    pcd_vals = [np.random.uniform(0, 1, (pcd_chain_size, i)) for i in dbm.num_units]

    loss, updates, new_pcd_states = dbm.train_step(lr, vis, pcd_states, mf_k, pcd_k)
    if output_dir is not None:
        sample_imgs = dbm.sample_from_dbm(100, 1000)
    sess = utils.get_session()
    with sess.as_default():
        tf.initialize_all_variables().run()

        for i in range(num_epoch):
            np.random.shuffle(xs)
            t = time.time()
            loss_vals = np.zeros(num_batches)
            for b in range(num_batches):
                batch_xs = xs[b*batch_size : (b+1)*batch_size]
                feed_dict = {vis: batch_xs,
                             lr: utils.scheduled_lr(init_lr, i, num_epoch)}
                for key, val in zip(pcd_states, pcd_vals):
                    feed_dict[key] = val

                loss_val, _, pcd_vals = sess.run(
                    [loss, updates, new_pcd_states], feed_dict=feed_dict)
                loss_vals[b] = loss_val
            print 'Epoch: %d, Train Loss: %s' % (i, loss_vals.mean())
            print '\tTime took:', time.time() - t

            if output_dir is not None:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                imgs = sess.run(sample_imgs)
                img_path = os.path.join(output_dir, 'epoch%d-plot.png' % i)
                utils.vis_samples(imgs, 10, 10, (28, 28), img_path)


def train_with_decoder(dbm, xs, init_lr, num_epoch, batch_size,
                       mf_k, pcd_k, pcd_chain_size, output_dir, decoder_dir):
    num_batches = len(xs) / batch_size
    assert num_batches * batch_size == len(xs)

    vis = tf.placeholder(tf.float32, (None,) + xs.shape[1:], name='vis_input')
    lr = tf.placeholder(tf.float32, (), name='lr')

    pcd_states = [tf.placeholder(tf.float32, (pcd_chain_size, i)) for i in dbm.num_units]
    pcd_vals = [np.random.uniform(0, 1, (pcd_chain_size, i)) for i in dbm.num_units]

    loss, updates, new_pcd_states = dbm.train_step(lr, vis, pcd_states, mf_k, pcd_k)
    if output_dir is not None:
        sample_imgs = dbm.sample_from_dbm(100, 1000)
        output_dir = os.path.join(decoder_dir, output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    sess = utils.get_session()
    with sess.as_default():
        tf.initialize_all_variables().run()

        encoder, decoder = keras_auto_encoder.load_encoder_decoder(
            (32, 32, 3),
            keras_auto_encoder.deep_encoder1, os.path.join(decoder_dir, 'encoder'),
            keras_auto_encoder.deep_decoder1, os.path.join(decoder_dir, 'decoder')
        )

        for i in range(num_epoch):
            np.random.shuffle(xs)
            t = time.time()
            loss_vals = np.zeros(num_batches)
            for b in range(num_batches):
                batch_xs = xs[b*batch_size : (b+1)*batch_size]
                feed_dict = {vis: batch_xs,
                             lr: utils.scheduled_lr(init_lr, i, num_epoch)}
                for key, val in zip(pcd_states, pcd_vals):
                    feed_dict[key] = val

                loss_val, _, pcd_vals = sess.run(
                    [loss, updates, new_pcd_states], feed_dict=feed_dict)
                loss_vals[b] = loss_val
            print 'Epoch: %d, Train Loss: %s' % (i+1, loss_vals.mean())
            print '\tTime took:', time.time() - t

            if (i+1) % 10 == 0 and output_dir is not None:
                saver = tf.train.Saver()
                save_path = saver.save(
                    sess, os.path.join(output_dir, 'epoch%d.ckpt' % (i+1)))
                print '\tModel saved to:', save_path
                imgs = sess.run(sample_imgs)
                img_path = os.path.join(output_dir, 'epoch%d-plot.png' % (i+1))
                decoder_input_shape = decoder.get_input_shape_at(0)[1:]
                imgs = imgs.reshape((-1,) + decoder_input_shape)
                imgs = decoder.predict(imgs)
                utils.vis_cifar10(imgs, 10, 10, img_path)



if __name__ == '__main__':
    lr = 0.001
    num_epoch = 200
    batch_size = 100
    mf_k = 10
    pcd_k = 5
    pcd_chain_size = 100
    output_dir = 'dbm-500-500-lr1e-3'

    # (train_xs, _), _, _ = cPickle.load(file('mnist.pkl', 'rb'))

    decoder_dir = 'old_noise_deep_model1'
    dataset = os.path.join(decoder_dir, 'encoded_cifar10.pkl')
    train_xs = cPickle.load(file(dataset, 'rb'))
    num_imgs = train_xs.shape[0]
    train_xs = train_xs.reshape(num_imgs, -1)
    print train_xs.shape


    batch_size = 20
    dbm = DBM([640, 500, 500], output_dir)
    train_with_decoder(dbm, train_xs, lr, num_epoch, batch_size,
                       mf_k, pcd_k, pcd_chain_size, output_dir, decoder_dir)

    # train(dbm, train_xs, lr, num_epoch, batch_size,
    #       mf_k, pcd_k, pcd_chain_size, output_dir)
