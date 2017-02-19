from dem import DEM
import numpy as np


def find_nearest_z_data(encoder, xs, z_sample):
    zs = encoder.predict(xs)
    z_data = np.zeros_like(z_sample)
    zz_distance = np.zeros(len(z_sample))
    for i in range(len(z_sample)):
        z_ref = z_sample[i]
        min_dis = float('inf')
        z_target = None
        for z in zs:
            dis = np.linalg.norm(z_ref-z)
            if dis < min_dis:
                z_target = z
                min_dis = dis
        zz_distance[i] = min_dis
        z_data[i] = z_target
    return z_data, zz_distance


if __name__ == '__main__':
    from dataset_wrapper import Cifar10Wrapper
    from rbm import RBM
    from autoencoder import AutoEncoder
    from dem_trainer import DEMTrainer
    import cifar10_ae
    import gibbs_sampler
    import utils

    import keras.backend as K
    import os
    import h5py
    import numpy as np

    np.random.seed(66699)
    sess = utils.create_session()
    K.set_session(sess)

    dataset = Cifar10Wrapper.load_default()
    ae_folder = 'prod/cifar10_ae3_relu_6/'
    encoder_weights_file = os.path.join(ae_folder, 'encoder.h5')
    decoder_weights_file = os.path.join(ae_folder, 'decoder.h5')
    rbm_params_file = os.path.join(
        ae_folder, 'ptrbm_scheme1/ptrbm_hid2000_lr0.001_pcd25/epoch_500_rbm.h5')

    # encoder_weights_file = '/home/hhu/Developer/dem/prod/cifar10_ae3_relu_6/test_ae_fe_const_balance/epoch_500_encoder.h5'
    # decoder_weights_file = encoder_weights_file.replace('encoder.', 'decoder.')
    # rbm_params_file = encoder_weights_file.replace('encoder.', 'rbm.')

    dem = DEM.load_from_param_files(dataset.x_shape, cifar10_ae.RELU_MAX,
                                    cifar10_ae.encode, encoder_weights_file,
                                    cifar10_ae.decode, decoder_weights_file,
                                    rbm_params_file)
    sampler_generator = gibbs_sampler.create_sampler_generator(
        dem.rbm, None, 64, 10000)
    output_dir = encoder_weights_file.rsplit('/', 1)[0]
    dem_trainer = DEMTrainer(sess, dataset, dem, utils.vis_cifar10, output_dir)

    z_sample = dem_trainer._draw_samples(sampler_generator())
    z_data, distance = find_nearest_z_data(dem.encoder, dataset.train_xs, z_sample)
    dem_trainer._save_samples(z_sample, encoder_weights_file+'.z_sample.png')
    dem_trainer._save_samples(z_data, encoder_weights_file+'.z_data.png')
    with open(encoder_weights_file+'.zz_distance.txt', 'w') as f:
        print >>f, distance
        for zd, zs in zip(z_data, z_sample):
            print >>f, list(zd[:20])
            print >>f, list(zs[:20])
            print >>f, ''
