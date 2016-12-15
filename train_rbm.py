import numpy as np
import cPickle
import os
import time
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import utils
import keras_auto_encoder# as ae
import vae
import keras_utils


def load_model(sess, model_path):
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print 'Model loaded from:', model_path


def train(rbm, dataset, lr, num_epoch, batch_size, use_pcd, cd_k, output_dir,
          pcd_chain_size=None, decoder_dir=None, init_with_test=False, subclass=None):
    train_xs, _ = dataset.get_subset('train', subclass)
    num_batches = len(train_xs) / batch_size
    assert num_batches * batch_size == len(train_xs)

    # graph related definitions
    ph_vis = tf.placeholder(tf.float32, [None]+rbm.vis_shape, name='vis_input')
    ph_lr = tf.placeholder(tf.float32, (), name='lr')
    if use_pcd:
        if not pcd_chain_size:
            pcd_chain_size = batch_size
        pcd_chain_shape = [pcd_chain_size] + rbm.vis_shape
        ph_pcd_chain = tf.placeholder(tf.float32, pcd_chain_shape, name='pcd_chain')
        pcd_chain_vis = np.random.uniform(size=pcd_chain_shape).astype(np.float32)
    else:
        ph_pcd_chain = None

    # Build the graph
    loss, cost, new_vis = rbm.get_loss_updates(ph_lr, ph_vis, ph_pcd_chain, cd_k)
    opt = tf.train.GradientDescentOptimizer(ph_lr)
    train_step = opt.minimize(cost)
    # Build sample generation part
    if output_dir is not None:
        num_samples = 100
        init_shape = [num_samples] + rbm.vis_shape
        if init_with_test:
            test_xs, _ = dataset.get_subset('test', subclass)
            num_steps = 1
        else:
            num_steps = 1000
        ph_sample_init = tf.placeholder(tf.float32, init_shape, name='sample_input')
        gen_samples = rbm.sample_from_rbm(num_steps, num_samples, ph_sample_init)
        output_dir = os.path.join(decoder_dir, output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    sess = utils.get_session()
    with sess.as_default():
        tf.initialize_all_variables().run()
        decoder = keras_utils.load_coder(dataset.latent_shape, eval(dataset.decoder),
                                         os.path.join(decoder_dir, 'decoder'))
        for i in range(num_epoch):
            t = time.time()
            np.random.shuffle(train_xs)
            loss_vals = np.zeros(num_batches)
            for b in range(num_batches):
                batch_xs = train_xs[b * batch_size:(b+1) * batch_size]
                if use_pcd:
                    loss_vals[b], _, pcd_chain_vis = sess.run(
                        [loss, train_step, new_vis],
                        feed_dict={ph_vis: batch_xs, ph_lr: lr,
                                   ph_pcd_chain: pcd_chain_vis})
                else:
                    loss_vals[b], _ = sess.run(
                            [loss,train_step], feed_dict={ph_vis: batch_xs, ph_lr: lr })

            print 'Epoch %d, Train Loss: %s' % (i, loss_vals.mean())
            print '\tTime took:', time.time() - t
            if (i+1) % 10 == 0 and output_dir is not None:
                # saver = tf.train.Saver()
                # save_path = saver.save(
                #     sess, os.path.join(output_dir, 'epoch%d.ckpt' % (i+1)))
                # print '\tModel saved to:', save_path

                # Generate samples
                img_path = os.path.join(output_dir, 'epoch%d-plot.png' % (i+1))
                if init_with_test:
                    init = test_xs[:num_samples]
                else:
                    init = np.random.normal(0, 1, init_shape).astype(np.float32)
                prob_imgs, _ = sess.run(
                    gen_samples, feed_dict={ph_sample_init: init})
                if dataset.scaled:
                    prob_imgs = prob_imgs * dataset.scale
                if dataset.normalized:
                    prob_imgs = prob_imgs * dataset.std + dataset.mean
                if decoder_dir is not None:
                    prob_imgs = prob_imgs.reshape((-1,) + dataset.latent_shape)
                    prob_imgs = decoder.predict(prob_imgs)
                utils.vis_cifar10(prob_imgs, 10, 10, img_path)
