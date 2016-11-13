import numpy as np
import cPickle
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from rbm import RBM
from crbm import CRBM
import train_rbm
from dbm import DBM


if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print 'usage: python conv.py pcd/cd cd-k [output_dir]'
        sys.exit()
    else:
        use_pcd = (sys.argv[1] == 'pcd')
        cd_k = int(sys.argv[2])
        output_dir = None if len(sys.argv) == 3 else sys.argv[3]

    (train_xs, _), _, _ = cPickle.load(file('mnist.pkl', 'rb'))
    train_xs = train_xs.reshape((-1, 28, 28, 1))
    batch_size = 20
    lr = 0.001 if use_pcd else 0.1

    dbm = DBM([28, 28, 1])
    dbm.add_conv_layer((12, 12, 1, 64), (2, 2), 'VALID', 'conv1')
    dbm.add_conv_layer((5, 5, 64, 128), (4, 4), 'VALID', 'conv2')
    # dbm.add_conv_layer((5, 5, 128, 128), (1, 1), 'SAME', 'conv3')
    # dbm.add_conv_layer((5, 5, 128, 128), (2, 2), 'SAME', 'conv4')
    dbm.add_fc_layer(500, 'fc1')
    # dbm.add_fc_layer(500, 'fc2')
    # dbm.add_fc_layer(1000, 'fc3')
    dbm.print_network()

    train_rbm.train(dbm, train_xs, lr, 40, batch_size, use_pcd, cd_k, output_dir)
