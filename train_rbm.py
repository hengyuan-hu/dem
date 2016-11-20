import numpy as np
import cPickle
import os
import time
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import utils


def load_model(sess, model_path):
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print 'Model loaded from:', model_path


def train(rbm, train_xs, lr, num_epoch, batch_size, use_pcd, cd_k, output_dir,
          pcd_chain_size=None, img_mean=0, img_std=1):
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
                if use_pcd:
                    loss_vals[b], _, persistent_vis_value = sess.run(
                        [loss, train_step, new_vis],
                        feed_dict={ph_vis: batch_xs,
                                   ph_lr: lr,
                                   persistent_vis_holder: persistent_vis_value})
                else:
                    loss_vals[b], _ = sess.run(
                            [loss,train_step], feed_dict={ph_vis: batch_xs, ph_lr: lr })

            print 'Train Loss:', loss_vals.mean()
            print '\tTime took:', time.time() - t
            if output_dir is not None:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                saver = tf.train.Saver()
                save_path = saver.save(
                    sess, os.path.join(output_dir, '%s-epoch%d.ckpt' % (rbm.name, i)))
                print '\tModel saved to:', save_path

                # Generate samples
                # num_samples = 100
                # num_steps = 3000
                # init_shape = tuple([num_samples] + rbm.vis_shape)
                # init = np.random.normal(0, 1, init_shape).astype(np.float32)
                # gen_samples = rbm.sample_from_rbm(num_steps, num_samples, init)
                # prob_imgs, sampled_imgs = sess.run(gen_samples)
                # prob_imgs = prob_imgs * img_std + img_mean
                # img_path = os.path.join(output_dir, 'epoch%d-plot.png' % i)
                # imgs = prob_imgs.reshape(num_samples, -1)
                # utils.vis_samples(imgs, 10, 10, (28, 28), img_path)
