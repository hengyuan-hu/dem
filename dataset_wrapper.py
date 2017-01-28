"""Wrapper classes for original and encoded datasets."""
from keras.datasets import mnist, cifar10
import numpy as np
import matplotlib.pyplot as plt
import h5py
import utils


class DatasetWrapper(object):
    def __init__(self, train_xs, train_ys, test_xs, test_ys):
        self.train_xs = train_xs
        self.train_ys = train_ys
        self.test_xs = test_xs
        self.test_ys = test_ys
        self.x_shape = self.train_xs.shape[1:]

    @classmethod
    def load_from_h5(cls, h5_path):
        with h5py.File(h5_path, 'r') as hf:
            train_xs = np.array(hf.get('train_xs'))
            train_ys = np.array(hf.get('train_ys'))
            test_xs = np.array(hf.get('test_xs'))
            test_ys = np.array(hf.get('test_ys'))
        print 'Dataset loaded from %s' % h5_path
        return cls(train_xs, train_ys, test_xs, test_ys)

    @classmethod
    def load_default(cls):
        raise NotImplementedError

    def dump_to_h5(self, h5_path):
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('train_xs', data=self.train_xs)
            hf.create_dataset('train_ys', data=self.train_ys)
            hf.create_dataset('test_xs', data=self.test_xs)
            hf.create_dataset('test_ys', data=self.test_ys)
        print 'Dataset written to %s' % h5_path

    def reshape(self, new_shape):
        self.train_xs = self.train_xs.reshape((-1,) + new_shape)
        self.test_xs = self.test_xs.reshape((-1,) + new_shape)
        self.x_shape = self.train_xs.shape[1:]
        assert self.x_shape == new_shape

    def plot_data_dist(self, fig_path, num_bins=50):
        xs = np.vstack((self.train_xs, self.test_xs))
        if len(xs.shape) > 2:
            num_imgs = len(xs)
            xs = xs.reshape((num_imgs, -1))
        plt.hist(xs, num_bins)
        if fig_path:
            plt.savefig(fig_path)
            plt.close()
        else:
            plt.show()

    def get_subset(self, subset, subclass=None):
        """get a subset.

        subset: 'train' or 'test'
        subclass: name of the subclass of interest
        """
        xs = self.train_xs if subset == 'train' else self.test_xs
        ys = self.train_ys if subset == 'train' else self.test_ys
        assert len(xs) == len(ys)

        if subclass:
            idx = self.cls2idx[subclass]
            loc = np.where(ys == idx)[0]
            xs = xs[loc]
            ys = ys[loc]
        return xs, ys


class MnistWrapper(DatasetWrapper):
    @classmethod
    def load_default(cls):
        ((train_xs, train_ys), (test_xs, test_ys)) = mnist.load_data()
        train_xs = (train_xs / 255.0).reshape(-1, 28, 28, 1)
        test_xs = (test_xs / 255.0).reshape(-1, 28, 28, 1)
        return cls(train_xs, train_ys, test_xs, test_ys)


class Cifar10Wrapper(DatasetWrapper):
    idx2cls = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    cls2idx = {cls: idx for (idx, cls) in enumerate(idx2cls)}

    @classmethod
    def load_default(cls):
        ((train_xs, train_ys), (test_xs, test_ys)) = cifar10.load_data()
        train_xs = utils.preprocess_cifar10(train_xs)
        test_xs = utils.preprocess_cifar10(test_xs)
        return cls(train_xs, train_ys, test_xs, test_ys)

if __name__ == '__main__':
    mnist_dataset = MnistWrapper.load_default()
    # mnist_dataset.plot_data_dist(None)
    cifar10_dataset = Cifar10Wrapper.load_default()
    # cifar10_dataset.plot_data_dist(None)
