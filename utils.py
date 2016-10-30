import matplotlib.pyplot as plt
import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    log_device_placement=True
    return tf.Session(config=config)


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

