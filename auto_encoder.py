import numpy as np
import cPickle
import os, sys
import tensorflow as tf
import utils
import time


class ConvLayer(object):
    def __init__(self, filter_shape, strides, padding, name):
        """construct a conv layer.
        filter_shape: 4D shape of [filter_height, filter_width, in_channels, out_channels]
        """
        assert len(filter_shape) == 4
        assert len(strides) == 2
        assert padding == 'SAME' or padding == 'VALID'

        self.name = name
        self.filter_shape = filter_shape
        self.strides = [1, strides[0], strides[1], 1]
        self.padding = padding

        self.weights = tf.get_variable(
            name+'_w', shape=filter_shape,
            initializer=tf.random_normal_initializer(0, 0.01))
        self.vbias = tf.get_variable(
            name+'_vbias',
            shape=filter_shape[-2], initializer=tf.constant_initializer(0.0))
        self.hbias = tf.get_variable(
            name+'hbias',
            shape=filter_shape[-1], initializer=tf.constant_initializer(0.0))

    def compute_up(self, data):
        conv = tf.nn.conv2d(data, self.weights, self.strides, self.padding)
        conv = tf.nn.bias_add(conv, self.hbias)
        self.input_shape = data.get_shape().as_list()
        return tf.nn.relu(conv)

    def compute_down(self, data):
        deconv = tf.nn.conv2d_transpose(data, self.weights, self.input_shape,
                                        self.strides, self.padding)
        deconv = tf.nn.bias_add(deconv, self.vbias)
        return tf.nn.relu(deconv)


class AutoEncoder(object):
    def __init__(self, vis_shape, name):
        """Initialize a auto encoder.

        vis_shape: 3D shape of [in_height, in_width, in_channel]
        """
        self.vis_shape = vis_shape
        self.name = name if name else 'autoec'
        self.data_shapes = [vis_shape]
        self.layers = []

    def add_conv_layer(self, filter_shape, strides, padding):
        """add a conv layer.
        filter_shape: 3D shape of [filter_height, filter_width, out_channels]
        """
        assert len(filter_shape) == 3
        filter_shape.insert(2, self.data_shapes[-1][-1])
        self.layers.append(
            ConvLayer(filter_shape, strides, padding, 'conv_%d' % len(self.layers)))

        output_shape = utils.conv_output_shape(
            self.data_shapes[-1], filter_shape, strides, padding)
        self.data_shapes.append(output_shape)

    def reconstruct(self, vis):
        encoded = vis
        for layer in self.layers:
            encoded = layer.compute_up(encoded)

        decoded = encoded
        for layer in self.layers[::-1]:
            decoded = layer.compute_down(decoded)
        return decoded

    def get_loss(self, vis):
        recon = self.reconstruct(vis)
        num_dims = len(vis.get_shape().as_list())
        dims = range(num_dims)
        total_loss = tf.reduce_sum(tf.square(vis - recon), dims[1:])
        return tf.reduce_mean(total_loss)


def train(model, train_xs, lr, num_epoch, batch_size, output_dir):
    vis_shape = train_xs.shape[1:]    # shape of single image
    batch_shape = (batch_size,) + vis_shape
    num_batches = len(train_xs) / batch_size
    assert num_batches * batch_size == len(train_xs)

    # graph related definitions
    ph_vis = tf.placeholder(tf.float32, batch_shape, name='vis_input')
    ph_lr = tf.placeholder(tf.float32, (), name='lr')

    # Build the graph
    loss = model.get_loss(ph_vis)
    # opt = tf.train.GradientDescentOptimizer(ph_lr)
    opt = tf.train.AdamOptimizer()
    train_step = opt.minimize(loss)
        
    # start a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    with sess.as_default():
        train_writer = tf.train.SummaryWriter('./train', sess.graph)
        tf.initialize_all_variables().run()

        for i in range(num_epoch):
            t = time.time()
            np.random.shuffle(train_xs)
            loss_vals = np.zeros(num_batches)
            for b in range(num_batches):
                batch_xs = train_xs[b * batch_size:(b+1) * batch_size]
                loss_vals[b], _ = sess.run(
                    [loss, train_step], feed_dict={ph_vis: batch_xs, ph_lr: lr })
            print 'Train Loss:', loss_vals.mean()
            print '\tTime took:', time.time() - t
            if output_dir is not None:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                saver = tf.train.Saver()
                save_path = saver.save(
                    sess, os.path.join(output_dir, '%s-epoch%d.ckpt' % (rbm.name, i)))
                print '\tModel saved to:', save_path


if __name__ == '__main__':
    from keras.datasets import cifar10

    output_dir = None if len(sys.argv) == 1 else sys.argv[1]

    (train_xs, _), (_, _) = cifar10.load_data()
    train_xs, mean, std = utils.preprocess_cifar10(train_xs)
    train_xs = train_xs[:, :, :, :1]
    print train_xs.shape

    # (train_xs, _), _, _ = cPickle.load(file('mnist.pkl', 'rb'))
    # train_xs = train_xs.reshape(-1, 28, 28, 1)

    batch_size = 20
    lr = 0.01

    model = AutoEncoder(train_xs[0].shape, output_dir)
    model.add_conv_layer([5, 5, 96], [1, 1], 'SAME')
    model.add_conv_layer([5, 5, 96], [1, 1], 'SAME')
    model.add_conv_layer([5, 5, 96], [1, 1], 'SAME')

    print model.data_shapes
    train(model, train_xs, lr, 40, batch_size, output_dir)
