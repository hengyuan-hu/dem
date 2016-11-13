import numpy as np
import cPickle
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from rbm import GaussianRBM, RBM
from crbm import GaussianCRBM
import train_rbm
from dbm import DBM
from keras.datasets import cifar10
import keras
import utils


def vis_cifar10(imgs):
    imgs = imgs * COLOR_STD_RGB + COLOR_MEAN_RGB
    plt.imshow(imgs)


if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print 'usage: python fc.py pcd/cd cd-k [output_dir]'
        sys.exit()
    else:
        use_pcd = (sys.argv[1] == 'pcd')
        cd_k = int(sys.argv[2])
        output_dir = None if len(sys.argv) == 3 else sys.argv[3]

    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print "INFO: temporarily set 'image_dim_ordering' to 'tf'"

    (train_xs, _), (_, _) = cifar10.load_data()
    train_xs, mean, std = utils.preprocess_cifar10(train_xs)
    batch_size = 20
    pcd_chain_size = 100
    lr = 0.0001 if use_pcd else 1e-5

    train_xs = train_xs[:,:,:,:1]#.reshape(-1, 32*32)
    rbm = GaussianCRBM((32, 32, 1), (12, 12, 1, 256), (2, 2), 'VALID', output_dir, {})
    # rbm = GaussianRBM(32*32, 1000, output_dir)
    # dbm = DBM([32*32], output_dir)
    # dbm.add_fc_layer(500, 'fc1', use_gaussian=True)
    # dbm.add_fc_layer(500, 'fc2')
    # dbm.add_fc_layer(1000, 'fc3')
    # dbm.print_network()

    train_rbm.train(rbm, train_xs, lr, 40, batch_size, use_pcd, cd_k, output_dir,
                    pcd_chain_size=pcd_chain_size)# , mean, std)
