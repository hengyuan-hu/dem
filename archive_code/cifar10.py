from keras.datasets import cifar10
import numpy as np
import h5py
import matplotlib.pyplot as plt
import utils


IDX2CLS = ['airplane', 'automobile', 'bird',
           'cat', 'deer', 'dog', 'frog',
           'horse', 'ship', 'truck']


CLS2IDX = {cls: idx for (idx, cls) in enumerate(IDX2CLS)}


def _flattern(dataset):
    batch_size = dataset.shape[0]
    return dataset.reshape((batch_size, -1))

def _deflattern(dataset, shape):
    batch_size = dataset.shape[0]
    return dataset.reshape((batch_size,) + shape)


class Cifar10Wrapper(object):
    def __init__(self, decoder=None, train_xs=None, train_ys=None,
                 test_xs=None, test_ys=None, mean=None, std=None):
        """Wrapper for cifar10 dataset, or encoded latent_cifar10 dataset."""
        self.normalized = False
        self.scaled = False

        if decoder is None:
            self.is_latent = False
            ((self.train_xs, self.train_ys),
             (self.test_xs, self.test_ys)) = cifar10.load_data()
            self.train_xs = utils.preprocess_cifar10(self.train_xs)
            self.test_xs = utils.preprocess_cifar10(self.test_xs)
            self.mean = 0.0
            self.std = 1.0
        else:
            assert (train_xs is not None and train_ys is not None
                    and test_xs is not None and test_ys is not None)
            self.is_latent = True
            self.decoder = decoder
            assert train_xs.shape[1:] == test_xs.shape[1:]
            self.latent_shape = train_xs.shape[1:]

            self.train_xs = _flattern(train_xs)
            self.train_ys = train_ys
            self.test_xs = _flattern(test_xs)
            self.test_ys = test_ys
            if mean is None:
                self.mean = np.vstack((self.train_xs, self.test_xs)).mean()
                print 'computed dataset mean:', self.mean
            else:
                self.mean = mean
            if std is None:
                self.std = np.vstack((self.train_xs, self.test_xs)).std()
                print 'computed dataset std:', self.std
            else:
                self.std = std

    @classmethod
    def load_from_h5(cls, h5_path):
        with h5py.File(h5_path, 'r') as hf:
            train_xs = np.array(hf.get('train_xs'))
            train_ys = np.array(hf.get('train_ys'))
            test_xs = np.array(hf.get('test_xs'))
            test_ys = np.array(hf.get('test_ys'))
            mean = hf.attrs['mean']
            std = hf.attrs['std']
            decoder = hf.attrs['decoder']
        print 'Dataset loaded from %s' % h5_path
        return cls(decoder, train_xs, train_ys, test_xs, test_ys, mean, std)

    def dump_to_h5(self, h5_path):
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('train_xs',
                              data=_deflattern(self.train_xs, self.latent_shape))
            hf.create_dataset('train_ys', data=self.train_ys)
            hf.create_dataset('test_xs',
                              data=_deflattern(self.test_xs, self.latent_shape))
            hf.create_dataset('test_ys', data=self.test_ys)
            hf.attrs['mean'] = self.mean
            hf.attrs['std'] = self.std
            hf.attrs['decoder'] = self.decoder
            # hf.create_dataset('mean_std', data=np.array([self.mean, self.std]))
        print 'Dataset written to %s' % h5_path

    def plot_distribution(self, fig_path):
        xs = np.vstack((self.train_xs, self.test_xs))
        plt.hist(xs, 50)
        plt.savefig(fig_path)
        plt.close()

    def normalize(self):
        self.train_xs = (self.train_xs - self.mean) / self.std
        self.test_xs = (self.test_xs - self.mean) / self.std
        self.normalized = True

    def scale(self):
        """Scale down to [0, 1]."""
        self.scale = max(self.train_xs.max(), self.test_xs.max())
        print 'Scaling dataset, scale: %s' % (self.scale)
        self.train_xs = self.train_xs / self.scale
        self.test_xs = self.test_xs / self.scale
        self.scaled = True

    def get_subset(self, subset, subclass):
        """get a subset.

        subset: 'train' or 'test'
        subclass: one of the 10 class in cifar10

        """
        xs = self.train_xs if subset == 'train' else self.test_xs
        ys = self.train_ys if subset == 'train' else self.test_ys
        assert len(xs) == len(ys)

        if subclass:
            idx = CLS2IDX[subclass]
            loc = np.where(ys == idx)[0]
            xs = xs[loc]
            ys = ys[loc]
        return xs, ys
