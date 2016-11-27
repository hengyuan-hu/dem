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


if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print 'usage: python train_auto_rbm.py pcd/cd cd-k [output_dir]'
        sys.exit()
    else:
        use_pcd = (sys.argv[1] == 'pcd')
        cd_k = int(sys.argv[2])
        output_dir = None if len(sys.argv) == 3 else sys.argv[3]

    decoder_dir = 'vae2'
    dataset = os.path.join(decoder_dir, 'bird.pkl')
    train_xs, zm_m, zm_std = cPickle.load(file(dataset, 'rb'))
    num_imgs = train_xs.shape[0]
    # train_xs = train_xs.reshape(num_imgs, -1)

    print train_xs.shape, train_xs.mean(), train_xs.std()
    batch_size = 20
    lr = 0.001 if use_pcd else 0.1

    rbm = GaussianRBM(train_xs[0].size, 1000, output_dir)

    train(rbm, train_xs, lr, 500, batch_size, use_pcd, cd_k, output_dir,
          pcd_chain_size=100, decoder_dir=decoder_dir, mean=zm_m, std=zm_std)
