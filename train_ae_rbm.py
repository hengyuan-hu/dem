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
import keras_auto_encoder
from cifar10 import Cifar10Wrapper


if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print 'usage: python train_auto_rbm.py pcd/cd cd-k [output_dir]'
        sys.exit()
    else:
        use_pcd = (sys.argv[1] == 'pcd')
        cd_k = int(sys.argv[2])
        output_dir = None if len(sys.argv) == 3 else sys.argv[3]
        init_with_test = not use_pcd

    # decoder_dir = 'relu_deep_model1_relu_6'
    decoder_dir = 'noise_deep_model2'
    dataset = Cifar10Wrapper.load_from_h5(
        os.path.join(decoder_dir, 'encoded_cifar10.h5'))
    # print '>>>>>'
    # dataset.scale()

    batch_size = 100
    lr = 0.001 if use_pcd else 0.1

    # rbm = GaussianRBM(dataset.train_xs[0].size, 1000, output_dir)
    rbm = RBM(dataset.train_xs[0].size, 2000, output_dir)

    train(rbm, dataset, lr, 500, batch_size, use_pcd, cd_k, output_dir,
          pcd_chain_size=100, decoder_dir=decoder_dir, init_with_test=init_with_test)
