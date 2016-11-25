import tensorflow as tf
import train_rbm
import utils
from rbm import RBM
import cPickle


if __name__ == '__main__':
    output_dir = 'rbm-1000-pcd1'
    model_file = 'noise_deep_model1_tuned_rbm/%s/epoch500.ckpt' % output_dir
    sess = utils.get_session()

    input_dim = 640
    rbm = RBM(input_dim, 1000, output_dir)
    ph_vis = tf.placeholder(tf.float32, [100, input_dim], name='vis_input')
    ph_lr = tf.placeholder(tf.float32, (), name='lr')

    loss, cost, new_vis = rbm.get_loss_updates(ph_lr, ph_vis, None, 1)
    opt = tf.train.GradientDescentOptimizer(ph_lr)
    train_step = opt.minimize(cost)


    with sess.as_default():
        tf.initialize_all_variables().run()
        train_rbm.load_model(sess, model_file)
        params = tf.get_collection(tf.GraphKeys.VARIABLES)

        params_val = sess.run(params)
        for p in params_val:
            print p.shape
    output_file = model_file + '_params.pkl'
    cPickle.dump(params_val,open(output_file, 'wb'))
    print 'params written into %s' % output_file
