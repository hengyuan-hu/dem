import numpy as np
import cPickle
import os
import time
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import utils
import keras_auto_encoder


def load_model(sess, model_path):
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print 'Model loaded from:', model_path


def train(rbm, train_xs, lr, num_epoch, batch_size, use_pcd, cd_k, output_dir,
          pcd_chain_size=None, decoder_dir=None):
    vis_shape = train_xs.shape[1:]    # shape of single image
    batch_shape = (batch_size,) + vis_shape
    num_batches = len(train_xs) / batch_size
    assert num_batches * batch_size == len(train_xs)

    # graph related definitions
    ph_vis = tf.placeholder(tf.float32, batch_shape, name='vis_input')
    ph_lr = tf.placeholder(tf.float32, (), name='lr')
    if use_pcd:
        if not pcd_chain_size:
            pcd_chain_size = batch_size
        pcd_chain_shape = (pcd_chain_size,) + vis_shape
        persistent_vis_holder = tf.placeholder(
            tf.float32, pcd_chain_shape, name='pst_vis_holder')
        persistent_vis_value = np.random.uniform(size=pcd_chain_shape).astype(np.float32)
    else:
        persistent_vis_holder = None

    # Build the graph
    loss, cost, new_vis = rbm.get_loss_updates(ph_lr, ph_vis, persistent_vis_holder, cd_k)
    opt = tf.train.GradientDescentOptimizer(ph_lr)
    train_step = opt.minimize(cost)
    # Build sample generation part
    if output_dir is not None:
        num_samples = 100
        num_steps = 1000
        init_shape = tuple([num_samples] + rbm.vis_shape)
        ph_sample_init = tf.placeholder(tf.float32, init_shape, name='sample_input')
        gen_samples = rbm.sample_from_rbm(num_steps, num_samples, ph_sample_init)
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
            t = time.time()
            np.random.shuffle(train_xs)
            loss_vals = np.zeros(num_batches)
            for b in range(num_batches):
                batch_xs = train_xs[b * batch_size:(b+1) * batch_size]
                if use_pcd:
                    loss_vals[b], _, persistent_vis_value = sess.run(
                        [loss, train_step, new_vis],
                        feed_dict={ph_vis: batch_xs, ph_lr: lr,
                                   persistent_vis_holder: persistent_vis_value})
                else:
                    loss_vals[b], _ = sess.run(
                            [loss,train_step], feed_dict={ph_vis: batch_xs, ph_lr: lr })

            print 'Train Loss:', loss_vals.mean()
            print '\tTime took:', time.time() - t
            if (i+1) % 10 == 0 and output_dir is not None:
                saver = tf.train.Saver()
                save_path = saver.save(
                    sess, os.path.join(output_dir, 'epoch%d.ckpt' % (i+1)))
                print '\tModel saved to:', save_path

                # Generate samples
                # num_samples = 100
                # num_steps = 1000
                img_path = os.path.join(output_dir, 'epoch%d-plot.png' % (i+1))
                # init_shape = tuple([num_samples] + rbm.vis_shape)
                init = np.random.normal(0, 1, init_shape).astype(np.float32)

                # gen_samples = rbm.sample_from_rbm(num_steps, num_samples, init)
                prob_imgs, _ = sess.run(
                    gen_samples, feed_dict={ph_sample_init: init})
                if decoder_dir is not None:
                    decoder_input_shape = decoder.get_input_shape_at(0)[1:]
                    # print decoder_input_shape
                    # print type(decoder_input_shape)
                    prob_imgs = prob_imgs.reshape((-1,) + decoder_input_shape)
                    prob_imgs = decoder.predict(prob_imgs)
                utils.vis_cifar10(prob_imgs, 10, 10, img_path)
                # prob_imgs = prob_imgs * img_std + img_mean
                # img_path = os.path.join(output_dir, 'epoch%d-plot.png' % i)
                # imgs = prob_imgs.reshape(num_samples, -1)
                # utils.vis_samples(imgs, 10, 10, (28, 28), img_path)
