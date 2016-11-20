import numpy as np
import cPickle
import os, sys
import time
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import utils
from rbm import RBM, GaussianRBM
from train_rbm import *



if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print 'usage: python train_auto_rbm.py pcd/cd cd-k [output_dir]'
        sys.exit()
    else:
        use_pcd = (sys.argv[1] == 'pcd')
        cd_k = int(sys.argv[2])
        output_dir = None if len(sys.argv) == 3 else sys.argv[3]

    train_xs = cPickle.load(file('noise_deep_encoder1_encoded_cifar10.pkl', 'rb'))
    num_imgs = train_xs.shape[0]
    train_xs = train_xs.reshape(num_imgs, -1)

    print train_xs.shape
    batch_size = 100
    lr = 0.0001 if use_pcd else 0.1

    # train_xs /= train_xs.max()

    rbm = RBM(train_xs[0].size, 500, output_dir)
    # rbm = GaussianRBM(train_xs[0].size, 2000, output_dir)

    train(rbm, train_xs, lr, 100, batch_size, use_pcd, cd_k, output_dir)
