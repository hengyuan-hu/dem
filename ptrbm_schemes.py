import os
import numpy as np

import keras.backend as K
import utils
from utils import TrainConfig
from rbm_pretrain import RBMPretrainer, pretrain


# from dem_trainer import DEMTrainer
from dataset_wrapper import Cifar10Wrapper
from rbm import RBM# , GibbsSampler
from autoencoder import AutoEncoder
import cifar10_ae
import utils



TRAIN_SCHEMES = {
    'ptrbm_scheme0': {
        'num_hid': 2000,
        'force_retrain': False,
        'train_configs':[
            TrainConfig(lr=0.1, batch_size=100, num_epoch=100, use_pcd=False, cd_k=1),
            TrainConfig(lr=0.05, batch_size=100, num_epoch=100, use_pcd=False, cd_k=3),
            TrainConfig(lr=0.01, batch_size=100, num_epoch=200, use_pcd=False, cd_k=10),
            TrainConfig(lr=0.001, batch_size=100, num_epoch=500, use_pcd=True, cd_k=1),
            TrainConfig(lr=0.001, batch_size=100, num_epoch=500, use_pcd=True, cd_k=5),
        ]},
    'ptrbm_scheme1': {
        'num_hid': 2000,
        'force_retrain': False,
        'train_configs':[
            TrainConfig(lr=0.1, batch_size=100, num_epoch=100, use_pcd=False, cd_k=1),
            TrainConfig(lr=0.05, batch_size=100, num_epoch=100, use_pcd=False, cd_k=5),
            TrainConfig(lr=0.02, batch_size=100, num_epoch=200, use_pcd=False, cd_k=10),
            TrainConfig(lr=0.01, batch_size=100, num_epoch=500, use_pcd=False, cd_k=25),
            TrainConfig(lr=0.005, batch_size=100, num_epoch=500, use_pcd=True, cd_k=5),
            TrainConfig(lr=0.002, batch_size=100, num_epoch=500, use_pcd=True, cd_k=10),
            TrainConfig(lr=0.001, batch_size=100, num_epoch=500, use_pcd=True, cd_k=25),
        ]},
    'ptrbm_scheme2':  {
        'num_hid': 4000,
        'force_retrain': False,
        'train_configs': [
            TrainConfig(lr=0.1, batch_size=100, num_epoch=100, use_pcd=False, cd_k=1),
            TrainConfig(lr=0.05, batch_size=100, num_epoch=200, use_pcd=False, cd_k=5),
            TrainConfig(lr=0.01, batch_size=100, num_epoch=500, use_pcd=False, cd_k=10),
            TrainConfig(lr=0.005, batch_size=100, num_epoch=500, use_pcd=True, cd_k=5),
            TrainConfig(lr=0.001, batch_size=100, num_epoch=500, use_pcd=True, cd_k=10),
            TrainConfig(lr=0.001, batch_size=100, num_epoch=500, use_pcd=True, cd_k=25),
        ]}
}


if __name__ == '__main__':
    np.random.seed(666)
    sess = utils.create_session()
    K.set_session(sess)

    ae_folder = 'prod/cifar10_ae2_relu_%d' % cifar10_ae.RELU_MAX
    ae = AutoEncoder(Cifar10Wrapper.load_default(),
                     cifar10_ae.encode, cifar10_ae.decode,
                     cifar10_ae.RELU_MAX, ae_folder)
    ae.build_models(ae_folder) # load model

    encoded_dataset = Cifar10Wrapper.load_from_h5(
        os.path.join(ae_folder, 'encoded_cifar10.h5'))
    assert len(encoded_dataset.x_shape) == 1

    name = 'ptrbm_scheme2'
    scheme = TRAIN_SCHEMES[name]
    output_folder = os.path.join(ae_folder, name)
    if os.path.exists(output_folder) and not scheme['force_retrain']:
        print '%s exists, skip training.' % name
        exit()
    print 'Training in:', output_folder
    rbm = RBM(encoded_dataset.x_shape[0], scheme['num_hid'], None)
    train_configs = scheme['train_configs']
    utils.log_train_configs(train_configs, output_folder)
    for config in train_configs:
        rbm_dir = pretrain(sess, rbm, encoded_dataset, ae.decoder,
                           config, utils.vis_cifar10, output_folder)
        rbm.save_model(sess, rbm_dir, 'epoch_%d_' % config.num_epoch)