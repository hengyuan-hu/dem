import keras.backend as K

import tensorflow as tf
import numpy as np
import os
import utils
import math
import time


class DEMTrainer(object):
    def __init__(self, sess, dataset, dem, sampler, sampler_generator, vis_fn):
        self.dataset = dataset
        self.dem = dem
        self.sampler = sampler
        self.sampler_generator = sampler_generator # TODO: factor out as train's param?
        self.vis_fn = vis_fn
        self.x_data = tf.placeholder(tf.float32, [None]+list(self.x_shape))
        self.x_model = tf.placeholder(tf.float32, [None]+list(self.x_shape))
        self.sess = sess
        self.log = []

    @property
    def x_shape(self):
        return self.dataset.x_shape

    def train(self, lr, num_epoch, batch_size, folder):
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        train_xs = self.dataset.train_xs
        num_batches = int(math.ceil(len(train_xs) / float(batch_size)))

        loss, cost = self.dem.loss_and_cost(self.x_data, self.x_model)
        opt = tf.train.GradientDescentOptimizer(lr)
        # opt = tf.train.AdamOptimizer(lr)
        train_step = opt.minimize(cost)
        if hasattr(self.sampler, 'samples'):
            # TODO: use a Sampler base class to improve this
            # use persistent chains
            sample_op, sampler_updates = self.sampler.sample()
        else:
            # does not use persistent chains
            print '>>>>>>>> using cd, not pcd'
            sample_op, sampler_updates = self.sampler.sample(self.x_data)

        # prevent tf.init from resetting encoder
        # K.tensorflow_backend._initialize_variables()
        utils.initialize_uninitialized_variables_by_keras()
        if hasattr(self.dem, 'ae'):
            l1_weights = self.dem.ae.encoder.layers[1].get_weights()
            print ('Before training: l1 weights sum: %s, bias sum: %s'
                   % (l1_weights[0].sum(), l1_weights[1].sum()))

        # helper graph for computing free energy
        fe_x = tf.placeholder(tf.float32, [None]+list(self.x_shape))
        fe_op = self.dem.free_energy(fe_x)

        for e in range(num_epoch):
            t = time.time()
            np.random.shuffle(train_xs)
            loss_vals = np.zeros(num_batches)
            for b in range(num_batches):
                x_data = train_xs[b*batch_size : (b+1)*batch_size]
                if not hasattr(self.sampler, 'samples'):
                    feed_dict = {self.x_data: x_data}
                else:
                    feed_dict = {}
                x_model, _ = self.sess.run([sample_op, sampler_updates], feed_dict)

                # fe_x_data = self.sess.run(fe_op, {fe_x: x_data})
                # fe_x_model = self.sess.run(fe_op, {fe_x: x_model})
                # if b % 10  == 0:
                #     print 'free energy\nx_data: %s;\nx_model: %s' % (
                #         fe_x_data[:5], fe_x_model[:5])

                feed_dict = {self.x_data: x_data, self.x_model: x_model}
                loss_vals[b], _ = self.sess.run([loss, train_step], feed_dict)
                # if b % 10 == 0:
                #     print '\tAccept rate:', self.sess.run([self.sampler.avg_accept_rate])
                #     print '\tStep size:', self.sess.run([self.sampler.stepsize])

            self.log.append('Epoch %d, Train Loss: %.4f' % (e+1, loss_vals.mean()))
            print self.log[-1]
            weights_sum = self.sess.run(self.dem.weights)
            weights_sum = weights_sum.sum()
            print '\tweights sum:', weights_sum
            # print '\tTime Taken: %ss' % (time.time() - t)
            # print '\tAccept rate:', self.sess.run([self.sampler.avg_accept_rate])
            # print '\tStep size:', self.sess.run([self.sampler.stepsize])

            if hasattr(self.dem, 'ae'):
                l1_weights = self.dem.ae.encoder.layers[1].get_weights()
                print ('\tl1 weights sum: %s, bias sum: %s'
                       % (l1_weights[0].sum(), l1_weights[1].sum()))

            if (e+1) % 5 == 0 and folder:
                samples = self._draw_samples()
                samples_path = os.path.join(folder, 'samples-epoch%d.png' % (e+1))
                chain_path = os.path.join(folder, 'pcd-chain-epoch%d.png' % (e+1))
                # print 'saving imgs'
                self._save_samples(samples, samples_path)
                # self.dem.save_model(self.sess, folder, 'epoch_%d_' % e)
                # self._save_samples(x_model, chain_path)

    def dump_log(self, folder):
        path = os.path.join(folder, 'dem_train.log')
        with open(path, 'w') as f:
            f.write('\n'.join(self.log))
            f.write('\n')

    def _draw_samples(self, burnin=1000):
        """Use a new sampler to draw samples from the trained model.
        """
        sampler = self.sampler_generator()
        # K.tensorflow_backend._initialize_variables()
        utils.initialize_uninitialized_variables_by_keras()

        sample_op, sampler_updates = sampler.sample()
        for i in range(burnin):
            self.sess.run([sample_op, sampler_updates])
            # if i % 10 == 0:
            #     print '\tAccept rate:', self.sess.run([sampler.avg_accept_rate])
            #     print '\tStep size:', self.sess.run([sampler.stepsize])

        samples, _ = self.sess.run([sample_op, sampler_updates])
        print 'in _draw_samples: samples min: %.4f, max: %.4f' \
            % (samples.min(), samples.max())
        return samples

    def _save_samples(self, samples, img_path):
        batch_size = len(samples)
        rows, cols = utils.factorize_number(batch_size)
        assert rows * cols == batch_size, '# of samples should be a square'
        self.vis_fn(samples, rows, cols, img_path)
