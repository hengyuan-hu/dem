import numpy as np
import cPickle
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from rbm import RBM
from crbm import CRBM
import train_rbm
from drbn import DRBN


if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print 'usage: python fc.py pcd/cd cd-k [output_dir]'
        sys.exit()
    else:
        use_pcd = (sys.argv[1] == 'pcd')
        cd_k = int(sys.argv[2])
        output_dir = None if len(sys.argv) == 3 else sys.argv[3]

    (train_xs, _), _, _ = cPickle.load(file('mnist.pkl', 'rb'))
    batch_size = 20
    lr = 0.001 if use_pcd else 0.1

    drbn = DRBN([784])
    drbn.add_fc_layer(500, 'fc1')
    drbn.add_fc_layer(500, 'fc2')
    drbn.add_fc_layer(1000, 'fc3')
    drbn.print_network()

    train_rbm.train(drbn, train_xs, lr, 40, batch_size, use_pcd, cd_k, output_dir)
