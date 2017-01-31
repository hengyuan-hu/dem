import time
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# import utils
import os
from scipy.signal import correlate, convolve


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def conv4d(x, weights, bias, output):
    # print 'called'
    assert len(x.shape) == 4 and len(output.shape) == 4
    batch_size, input_channel = x.shape[:2]
    output_batch_size, output_channel = output.shape[:2]
    num_filters, filter_channel = weights.shape[:2]
    assert batch_size == output_batch_size, '%d vs %d' % (batch_size, output_batch_size)
    assert output_channel == num_filters
    assert filter_channel == input_channel

    # func = convolve if true_conv else correlate
    for img_idx in range(batch_size):
        for c in range(output_channel):
            output[img_idx][c] = (correlate(x[img_idx], weights[c], mode='valid')
                                  + bias[c].reshape((1, 1, 1)))
            # if img_idx == 0 and c == 0:
            #     print output[img_idx][c]
            #     print bias[c].reshape((1, 1, 1))


class CRBM(object):
    def __init__(self, first_batch, filter_shape, load_params=None):
        # assert len(vis_shape) == 4
        assert len(filter_shape) == 4
        assert filter_shape[2] == filter_shape[3], 'square filter only'
        assert filter_shape[2] % 2 == 1, 'filter size must be odd'
        assert len(first_batch.shape) == 4

        self.batch_size = first_batch.shape[0]
        self.vis_shape = first_batch.shape # 4D shape (batch_size, channel, width, height)
        self.filter_shape = filter_shape # 4D shape (filter, channel, width, height)
        self.hid_shape = (self.batch_size,) + filter_shape[:1] + self.vis_shape[2:]
        self.pad_dim = (self.filter_shape[2] - 1) / 2
        # print 'vis_shape:', self.vis_shape, '; hid_shape:', self.hid_shape
        # self.num_hid = filter_shape[0]
        if load_params is not None:
            self.weights, self.vbias, self.hbias = load_rbm_params(load_params)
        else:
            self.weights = np.random.normal(0, 0.01, self.filter_shape)
            # bias are represented as vectors, not matrices
            # num of bias equals to num of channels
            self.vbias = np.zeros(self.vis_shape[1])
            self.hbias = np.zeros(self.hid_shape[1])

        if first_batch is None:
            self.persistent_hid = None
        else:
            self.persistent_hid = self.sample(self.compute_up(first_batch))

    # def _compute_vis_bias(self, train_x):
    #     p = train_x.mean(axis=0) + 0.0001
    #     return np.log(p) - np.log(1-p)

    def compute_up(self, vis):
        hid_p = np.zeros(self.hid_shape)
        # print 'hid shape:', hid_p.shape
        # print 'hbias shape', self.hbias.shape
        padded_vis = np.pad(vis, ((0,), (0,), (self.pad_dim,), (self.pad_dim,)),
                            'constant', constant_values=(0,))
        conv4d(padded_vis, self.weights, self.hbias, hid_p)
        hid_p = sigmoid(hid_p)
        return hid_p

    def compute_down(self, hid):
        vis_p = np.zeros(self.vis_shape)
        # print 'vis shape:', vis_p.shape
        # print 'vbias shape', self.vbias.shape

        weights = self.weights.swapaxes(0, 1)
        weights = weights[:, :, ::-1, ::-1]
        padded_hid = np.pad(hid, ((0,), (0,), (self.pad_dim,), (self.pad_dim,)),
                            'constant', constant_values=(0,))
        conv4d(padded_hid, weights, self.vbias, vis_p)
        vis_p = sigmoid(vis_p)
        return vis_p
        # vis_p = sigmoid(np.dot(hid, self.weights.T) + self.vbias)
        # return vis_p

    def sample(self, ps):
        samples = (ps > np.random.uniform(0, 1, ps.shape)).astype(np.float32, copy=False)
        return samples

    def sample_vhv(self, vis):
        hid = self.sample(self.compute_up(vis))
        new_vis_p = self.compute_down(hid)
        new_vis_samples = self.sample(new_vis_p)
        return new_vis_p, new_vis_samples

    def cd(self, k, vis):
        """Contrastive Divergence."""
        vis_p = vis
        vis_samples = vis
        for _ in range(k):
            hid_p = self.compute_up(vis_samples)
            hid_samples = self.sample(hid_p)
            vis_p = self.compute_down(hid_samples)
            vis_samples = self.sample(vis_p)
        return vis_p, vis_samples

    def pcd(self):
        recon_vis_p = self.compute_down(self.persistent_hid)
        recon_vis_samples = self.sample(recon_vis_p)
        recon_hid_p = self.compute_up(recon_vis_samples)
        self.persistent_hid = self.sample(recon_hid_p)
        return recon_vis_p, recon_hid_p

    def collect_stats(self, vis, hid):
        assert len(vis.shape) == 4 and len(hid.shape) == 4
        assert vis.shape[0] == hid.shape[0], 'batch size'
        batch_size = vis.shape[0]

        stats_w = np.zeros(self.filter_shape)
        padded_vis = np.pad(vis, ((0,), (0,), (self.pad_dim,), (self.pad_dim,)),
                            'constant', constant_values=(0,))
        conv_weights = hid.swapaxes(0, 1)
        conv_input = padded_vis.swapaxes(0, 1)
        stats_w = stats_w.swapaxes(0, 1)
        # print '>>>', conv_input.shape, conv_weights.shape
        conv4d(conv_input, conv_weights, np.zeros(conv_weights.shape[0]), stats_w)
        stats_w = stats_w.swapaxes(0, 1) / batch_size

        stats_vbias = vis.sum(axis=(2,3)).mean(axis=0)
        stats_hbias = hid.sum(axis=(2,3)).mean(axis=0)

        # print 'stats_w:', stats_w.shape, 'stats_vbias:', stats_vbias.shape, 'stats_hbias:', stats_hbias.shape
        return stats_w, stats_vbias, stats_hbias

    def compute_gradient(self, vis, cd_k):
        hid_p = self.compute_up(vis)
        pos_stats_w, pos_stats_vbias, pos_stats_hbias = self.collect_stats(vis, hid_p)

        if cd_k is None:
            recon_vis_p, recon_hid_p = self.pcd()
        else:
            recon_vis_p, _ = self.cd(cd_k, vis)
            recon_hid_p = self.compute_up(recon_vis_p)
        neg_stats_w, neg_stats_vbias, neg_stats_hbias = self.collect_stats(recon_vis_p, recon_hid_p)
        # negative_stats = self.collect_stats(recon_vis_p, recon_hid_p)
        # assert positive_stats.shape == self.weights.shape

        dw = pos_stats_w - neg_stats_w
        dvbias = pos_stats_vbias - neg_stats_vbias
        dhbias = pos_stats_hbias - neg_stats_hbias
        return dw, dvbias, dhbias

    def compute_l2_loss(self, vis):
        recon_vis_p, _ = self.cd(1, vis)
        loss = np.square(vis - recon_vis_p).sum()
        return loss

    def one_iter(self, vis, cd_k, lr):
        dw, dvbias, dhbias = self.compute_gradient(vis, cd_k)
        # derivatives are d(l)/d(.), so should use gradient ascent here
        self.weights += lr * dw
        self.vbias += lr * dvbias
        self.hbias += lr * dhbias
        return self.compute_l2_loss(vis)


def dump_rbm_params(file_name, rbm, for_tf=False):
    weights = rbm.weights
    if for_tf:
        weights = weights.swapaxes(1, 2)
        weights = weights.swapaxes(2, 3)

        weights = weights.swapaxes(0, 1)
        weights = weights.swapaxes(1, 2)
        weights = weights.swapaxes(2, 3)
    # print weights.shape
    # print rbm.vbias.shape
    # print rbm.hbias.shape
    # print weights
    params = {'weights': weights, 'vbias': rbm.vbias, 'hbias': rbm.hbias}
    cPickle.dump(params, file(file_name, 'wb'))


def load_rbm_params(file_name):
    params = cPickle.load(file(file_name, 'rb'))
    return [params['weights'], params['vbias'], params['hbias']]


def vis_weights(weights, rows, cols, neuron_shape, output_name=None):
    assert weights.shape[-1] == rows * cols
    f, axarr = plt.subplots(rows, cols)
    for r in range(rows):
        for c in range(cols):
            neuron_idx = r * cols + c
            weight_map = weights[:, neuron_idx].reshape(neuron_shape)
            axarr[r][c].imshow(weight_map, cmap='Greys')
            axarr[r][c].set_axis_off()
    f.subplots_adjust(hspace=0.2, wspace=0.2)
    if output_name is None:
        plt.show()
    else:
        plt.savefig(output_name)


# def sample_from_crbm(rbm, num_samples, num_snapshots, steps_per_snapshot, use_sample, prefix):
#     vis_p = np.random.normal(0, 1, (num_samples, 1, 28, 28))
#     image_data = np.zeros((29 * num_snapshots + 1, 29 * num_samples - 1), dtype='uint8')
#     for i in range(num_snapshots):
#         print i
#         for _ in range(steps_per_snapshot):
#             vis_p, vis_samples = rbm.sample_vhv(vis_p)
#         if use_sample:
#             img = vis_samples
#         else:
#             img = vis_p
#         current_snapshot = utils.tile_raster_images(
#             X=img, img_shape=(28, 28), tile_shape=(1, num_samples), tile_spacing=(1, 1))
#         image_data[29 * i:29 * i + 28, :] = current_snapshot
#         # snap_image = Image.fromarray(current_snapshot)
#         # snap_image.save('%s_step%d_sample.png' % (prefix, (i+1) * steps_per_snapshot))

#     if prefix is None:
#         plt.imshow(image_data, cmap='Greys',  interpolation='nearest')
#     else:
#         image = Image.fromarray(image_data)
#         image.save('%s_sample.png' % prefix)


if __name__ == '__main__':
    (train_xs, _), _, _ = cPickle.load(file('mnist.pkl', 'rb'))
    train_xs = train_xs.reshape((-1, 1, 28, 28))
    print 'training set shape:', train_xs.shape
    batch_size = 20
    num_epoch = 50
    iter_per_epoch = len(train_xs) / batch_size

    # print '>>>', train_xs[:batch_size].shape
    crbm = CRBM(train_xs[:batch_size], (200, 1, 5, 5))
    cd_k = None
    lr = 1e-5
    # cd_k = 1
    # lr = 1e-4

    # output_folder = 'pcd-lr1e-3-plot-prob'
    output_folder = 'crbm-200-pcd-1e-5'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    log_file = open(os.path.join(output_folder, 'loss.log'), 'w', 0)
    prefix_pattern = os.path.join(output_folder, 'epoch%d')
    param_saver_pattern = os.path.join(output_folder, 'epoch%d_params.pkl')
    epoch_per_snapshot_for_params = 1

    for e in range(num_epoch):
        loss = np.zeros(iter_per_epoch)
        start_time = time.time()
        for i in range(iter_per_epoch):
            vis = train_xs[i * batch_size: (i+1) * batch_size]
            loss[i] = crbm.one_iter(vis, cd_k, lr)            
            if (i+1) % 100 == 0:
                print 'time used:', time.time() - start_time
                start_time = time.time()
                print i, 'loss:', loss.sum() / (i+1)
        print 'epoch:', e+1, ', loss:', loss.mean()
        print >>log_file, 'epoch:', e+1, ', loss:', loss.mean()

        # sample_from_crbm(crbm, 5, 5, 30, use_sample=False, prefix=prefix_pattern % (e+1))

        if (e+1) % epoch_per_snapshot_for_params == 0:
            dump_rbm_params(param_saver_pattern % (e+1), crbm)

    log_file.close()
