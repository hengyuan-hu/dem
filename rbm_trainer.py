import os
import math
import time
import numpy as np
import tensorflow as tf
import utils


class RBMTrainer(object):
    def __init__(self, sess, dataset, rbm, vis_fn, output_dir):
        self.dataset = dataset
        self.rbm = rbm
        self.vis_fn = vis_fn
        self.sess = sess
        self.log = []
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    @property
    def x_shape(self):
        return [None] + list(self.dataset.x_shape)

    def train(self, train_config, sampler, sampler_generator):
        # define graph
        x_data_node = tf.placeholder(tf.float32, self.x_shape)
        x_model_node = tf.placeholder(tf.float32, self.x_shape)
        loss, cost = self.rbm.loss_and_cost(x_data_node, x_model_node)
        opt = tf.train.GradientDescentOptimizer(train_config.lr)
        train_step = opt.minimize(cost)
        # define sampler graph
        if sampler.is_persistent:
            print '>>> train with pcd'
            sample_op, sampler_updates = sampler.sample()
        else:
            print '>>> train with cd'
            sample_op, sampler_updates = sampler.sample(x_data_node)

        # prevent tf.init from resetting encoder
        utils.initialize_uninitialized_variables_by_keras()

        train_xs = self.dataset.train_xs
        num_batches = int(math.ceil(len(train_xs) / float(train_config.batch_size)))
        for e in range(train_config.num_epoch):
            t = time.time()
            np.random.shuffle(train_xs)
            loss_vals = np.zeros(num_batches)
            for b in range(num_batches):
                x_data = train_xs[b * train_config.batch_size
                                  :(b+1) * train_config.batch_size]
                feed_dict = {} if sampler.is_persistent else {x_data_node: x_data}
                x_model, _ = self.sess.run([sample_op, sampler_updates], feed_dict)
                feed_dict = {x_data_node: x_data, x_model_node: x_model}
                loss_vals[b], _ = self.sess.run([loss, train_step], feed_dict)

            self.log.append('Epoch %d, Train Loss: %.4f' % (e+1, loss_vals.mean()))
            print self.log[-1]
            print '\tTime Taken: %ss' % (time.time() - t)

            if (e+1) % 10 == 0 and self.output_dir:
                samples = self._draw_samples(sampler_generator())
                samples_path = os.path.join(
                    self.output_dir, 'samples-epoch%d.png' % (e+1))
                self._save_samples(samples, samples_path)
            if (e+1) % 100 == 0 and self.output_dir:
                self.rbm.save_model(self.sess, self.output_dir, 'epoch_%d_' % (e+1))
                self.dump_log()

    def dump_log(self, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
        path = os.path.join(output_dir, 'rbm_train.log')
        with open(path, 'w') as f:
            f.write('\n'.join(self.log))
            f.write('\n')

    def _draw_samples(self, sampler):
        """Use a new sampler to draw samples from the trained model.
        """
        assert sampler.is_persistent
        utils.initialize_uninitialized_variables_by_keras()

        sample_op, sampler_updates = sampler.sample()
        for _ in range(sampler.burnin):
            self.sess.run([sample_op, sampler_updates])

        samples, _ = self.sess.run([sample_op, sampler_updates])
        print 'in _draw_samples: samples min: %.4f, max: %.4f' \
            % (samples.min(), samples.max())
        return samples

    def _save_samples(self, samples, img_path):
        batch_size = len(samples)
        rows, cols = utils.factorize_number(batch_size)
        self.vis_fn(samples, rows, cols, img_path)
