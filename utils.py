import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


CIFAR10_COLOR_MEAN_RGB = np.array([125.3, 123.0, 113.9]).reshape(1, 1, 3)
CIFAR10_COLOR_STD_RGB  = np.array([63.0,  62.1,  66.7]).reshape(1, 1, 3)


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def sample_bernoulli(ps):
    rand_uniform = tf.random_uniform(tf.shape(ps), 0, 1)
    samples = tf.to_float(rand_uniform < ps)
    return samples


def scheduled_lr(base_lr, epoch, total_epoch):
    if epoch < 0.5 * total_epoch:
        return base_lr
    elif epoch < 0.75 * total_epoch:
        return base_lr * 0.1
    else:
        return base_lr * 0.01


def preprocess_cifar10(dataset):
    dataset = (dataset - CIFAR10_COLOR_MEAN_RGB) / CIFAR10_COLOR_STD_RGB
    return dataset


def vis_cifar10(imgs, rows, cols, output_name):
    imgs = imgs * CIFAR10_COLOR_STD_RGB + CIFAR10_COLOR_MEAN_RGB
    print imgs.min(), imgs.max()
    imgs = np.maximum(np.zeros(imgs.shape), imgs)
    imgs = np.minimum(np.ones(imgs.shape)*255, imgs)
    print imgs.shape
    imgs = imgs.astype(np.uint8)

    assert imgs.shape[0] == rows * cols
    f, axarr = plt.subplots(rows, cols, figsize=(32, 32))
    for r in range(rows):
        for c in range(cols):
            img = imgs[r * cols + c]
            axarr[r][c].imshow(img)
            axarr[r][c].set_axis_off()
    f.subplots_adjust(hspace=0.2, wspace=0.2)
    if output_name is None:
        plt.show()
    else:
        plt.savefig(output_name)
    plt.close()


def vis_samples(imgs, rows, cols, img_shape, output_name):
    return vis_weights(imgs.T, rows, cols, img_shape, output_name, 'Greys_r')


def vis_weights(weights, rows, cols, neuron_shape, output_name=None, cmap='Greys'):
    assert weights.shape[-1] == rows * cols
    f, axarr = plt.subplots(rows, cols, figsize=neuron_shape)
    for r in range(rows):
        for c in range(cols):
            neuron_idx = r * cols + c
            weight_map = weights[:, neuron_idx].reshape(neuron_shape)
            axarr[r][c].imshow(weight_map, cmap=cmap)
            axarr[r][c].set_axis_off()
    f.subplots_adjust(hspace=0.2, wspace=0.2)
    if output_name is None:
        plt.show()
    else:
        plt.savefig(output_name)
    plt.close()


def conv_output_length(input_length, filter_size, stride, border_mode, dilation=1):
    if input_length is None:
        return None
    assert border_mode in {'SAME', 'VALID'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'SAME':
        output_length = input_length
    elif border_mode == 'VALID':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


def conv_output_shape(input_shape, filter_shape, strides, padding):
    """Compute the output shape of a convolution operation on particular input.

    input_shape: 3D shape of [in_height, in_width, in_channel]
    filter_shape: 4D shape of [filter_height, filter_width, in_channels, out_channels]
    strides: 2D shape of [stride_height, stride_width]
    padding: string, 'SAME' or 'VALID'
    return: 3D shape of [out_height, out_width, out_channel]
    """
    assert len(input_shape) == 3, len(input_shape)
    assert len(filter_shape) == 4, len(filter_shape)
    assert input_shape[2] == filter_shape[2]
    assert len(strides) == 2, len(strides)
    assert padding == 'SAME' or 'VALID', padding
    out_height = conv_output_length(input_shape[0], filter_shape[0], strides[0], padding)
    out_width = conv_output_length(input_shape[1], filter_shape[1], strides[1], padding)
    return [out_height, out_width, filter_shape[3]]
