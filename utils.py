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

