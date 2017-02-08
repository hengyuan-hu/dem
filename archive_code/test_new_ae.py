from cifar10_ae import *
from autoencoder import AutoEncoder
from dataset_wrapper import Cifar10Wrapper
import keras.backend as K


def compare_dataset():
    d1 = Cifar10Wrapper.load_from_h5('prod/test_relu6/encoded_cifar10.h5')
    d2 = Cifar10Wrapper.load_from_h5('prod/cifar10_ae2_relu_6/encoded_cifar10.h5')

    return d1, d2


if __name__ == '__main__':
    K.set_session(utils.create_session())
    cifar10_dataset = Cifar10Wrapper.load_default()

    folder = 'prod/test_relu6'
    ae = AutoEncoder(cifar10_dataset, encode, decode, RELU_MAX, folder)
    ae.build_models(folder) # load previously trained ae

    # num_epoch = 2
    # lr_schedule = utils.generate_decay_lr_schedule(num_epoch, 0.1, 1)
    # ae.train(128, num_epoch, lr_schedule)
    # ae.save_models()
    # ae.test_models(utils.vis_cifar10)
    # ae.log()

    encoded_dataset = ae.encode(Cifar10Wrapper)
    # encoded_dataset.dump_to_h5(os.path.join(folder, 'encoded_cifar10.h5'))
    # encoded_dataset.plot_data_dist(os.path.join(folder, 'encoded_plot.png'))
