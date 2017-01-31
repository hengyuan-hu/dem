import keras.backend as K

from dataset_wrapper import MnistWrapper, Cifar10Wrapper
from rbm import RBM, GibbsSampler
from autoencoder import AutoEncoder
from rbm_pretrain import RBMPretrainer
import cifar10_ae
import utils
import os

if __name__ == '__main__':

    sess = utils.create_session()
    K.set_session(sess)

    ae_folder = 'relu_deep_model1_relu_6_reprod'
    ae = AutoEncoder(Cifar10Wrapper.load_default(),
                     cifar10_ae.encode, cifar10_ae.decode,
                     cifar10_ae.RELU_MAX, ae_folder)
    ae.build_models(ae_folder) # load model
    # ae.test_models(utils.vis_cifar10)
    # encoded_dataset = ae.encode(Cifar10Wrapper)
    # encoded_dataset.plot_data_dist(os.path.join(ae_folder, 'data_dist.png'))

    # dataset = MnistWrapper.load_from_h5('test/mnist_ae_relu_6/encoded_mnist.h5')
    encoded_dataset = Cifar10Wrapper.load_from_h5(
        os.path.join(ae_folder, 'encoded_cifar10.h5'))

    encoded_dataset.reshape((1024,))
    encoded_dataset.train_xs = encoded_dataset.train_xs / 6.0
    encoded_dataset.test_xs = encoded_dataset.test_xs / 6.0

    assert len(encoded_dataset.x_shape) == 1
    num_vis = encoded_dataset.x_shape[0]
    num_hid = 2000

    rbm = RBM(num_vis, num_hid, None)

    print '>>>> test_xs:', encoded_dataset.test_xs.min(), encoded_dataset.test_xs.max()

    def sampler_generator(cd_k=1, init_vals=encoded_dataset.test_xs[:100]):
        return GibbsSampler(init_vals, rbm, cd_k)

    batch_size = 100
    num_epoch = 100
    lr = 0.1
    cd_k = 1
    gibbs_sampler = sampler_generator(cd_k=cd_k, init_vals=None)

    trainer = RBMPretrainer(
        sess, encoded_dataset, rbm, ae.decoder,
        gibbs_sampler, sampler_generator, utils.vis_cifar10)

    rbm_folder = 'pretrained_rbm_hid_%d_lr_%s_batch_%d' % (num_hid, lr, batch_size)
    output_folder = os.path.join(ae_folder, rbm_folder)
    trainer.train(lr, 100, batch_size, output_folder)
    trainer.dump_log(output_folder)
