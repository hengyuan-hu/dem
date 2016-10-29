# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import math
import numpy as np
import cPickle

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import mnist_dataset
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.examples.tutorials.mnist import input_data

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 5000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 500, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 500, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', 1000, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

DATASET = 'pickle'

def inference(images, hidden1_units, hidden2_units, hidden3_units):
    """Build the MNIST model up to where it may be used for inference.

    Args:
        images: Images placeholder, from inputs().
        hidden1_units: Size of the first hidden layer.
        hidden2_units: Size of the second hidden layer.
        hidden3_units: Size of the third hidden layer.

    Returns:
        softmax_linear: Output tensor with the computed logits.
    """
    # Hidden 1
    with tf.variable_scope('hidden1'):
        weights = tf.get_variable('weights',
                                  shape = [IMAGE_PIXELS, hidden1_units],
                                  initializer=tf.truncated_normal_initializer(0, 0.01))
        biases = tf.get_variable('biases',
                                 shape = [hidden1_units],
                                 initializer=tf.constant_initializer(0.0))
        hidden1 = tf.nn.sigmoid(tf.matmul(images, weights) + biases)
        # Hidden 2
    with tf.variable_scope('hidden2'):
        weights = tf.get_variable('weights',
                                  shape = [hidden1_units, hidden2_units],
                                  initializer=tf.truncated_normal_initializer(0, 0.01))
        biases = tf.get_variable('biases',
                                 shape = [hidden2_units],
                                 initializer=tf.constant_initializer(0.0))
        hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, weights) + biases)
        # Hidden 3
    with tf.variable_scope('hidden3'):
        weights = tf.get_variable('weights',
                                  shape = [hidden2_units, hidden3_units],
                                  initializer=tf.truncated_normal_initializer(0, 0.01))
        biases = tf.get_variable('biases',
                                 shape = [hidden3_units],
                                 initializer=tf.constant_initializer(0.0))
        hidden3 = tf.nn.sigmoid(tf.matmul(hidden2, weights) + biases)
        # Linear
    with tf.variable_scope('softmax_linear'):
        weights = tf.get_variable('weights',
                                  shape = [hidden3_units, NUM_CLASSES],
                                  initializer=tf.truncated_normal_initializer(0, 0.01))
        biases = tf.get_variable('biases',
                                 shape = [NUM_CLASSES],
                                 initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(hidden3, weights) + biases
    return logits

def training(loss):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.

    Returns:
        train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    # optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    optimizer = tf.train.AdamOptimizer()

    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
        batch_size: The batch size will be baked into both placeholders.

    Returns:
        images_placeholder: Images placeholder.
        labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }

    Args:
        data_set: The set of images and labels, from mnist_dataset.read_data_sets()
        images_pl: The images placeholder, from placeholder_inputs().
        labels_pl: The labels placeholder, from placeholder_inputs().

    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.

    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
        mnist_dataset.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))

def run_training():
    """Train MNIST for a number of steps."""
    # Get the sets of images and labels for training, validation, and
    # test on MNIST.
    if DATASET == 'pickle':
        data_sets = mnist_dataset.read_data_sets(FLAGS.train_dir, data_dir='../mnist.pkl')
    elif DATASET == 'keras':
        data_sets = mnist_dataset.read_data_sets(FLAGS.train_dir, keras=True)
    else:
        data_sets = input_data.read_data_sets(FLAGS.train_dir)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the images and labels.
        images_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size)

        # Build a Graph that computes predictions from the inference model.
        logits = inference(images_placeholder,
                           FLAGS.hidden1,
                           FLAGS.hidden2,
                           FLAGS.hidden3)

        # Add to the Graph the Ops for loss calculation.
        loss = mnist.loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.merge_all_summaries()

        # Add the variable initializer Op.
        init = tf.initialize_all_variables()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        steps_per_epoch = data_sets.train.num_examples // FLAGS.batch_size
        # Start the training loop.
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(data_sets.train,
                                    images_placeholder,
                                    labels_placeholder)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss],
                                    feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % steps_per_epoch == 0:
                # Print status to stdout.
                print('Epoch %d: loss = %.2f (%.3f sec)' % (step/steps_per_epoch, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 10000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.train)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.test)

        # save weights
        with tf.variable_scope("hidden1",reuse=True):
            W = tf.get_variable("weights",[IMAGE_PIXELS, FLAGS.hidden1])
            b = tf.get_variable("biases",[FLAGS.hidden1])
            W_1, b_1 = W.eval(sess), b.eval(sess)
            #np.savetxt("weights/hidden1_W.csv", W_val, delimiter=",")
            #np.savetxt("weights/hidden1_b.csv", b_val, delimiter=",")

        with tf.variable_scope("hidden2",reuse=True):
            W = tf.get_variable("weights",[FLAGS.hidden1, FLAGS.hidden2])
            b = tf.get_variable("biases",[FLAGS.hidden2])
            W_2, b_2 = W.eval(sess), b.eval(sess)
            #np.savetxt("weights/hidden2_W.csv", W_val, delimiter=",")
            #np.savetxt("weights/hidden2_b.csv", b_val, delimiter=",")

        with tf.variable_scope("hidden3",reuse=True):
            W = tf.get_variable("weights",[FLAGS.hidden2, FLAGS.hidden3])
            b = tf.get_variable("biases",[FLAGS.hidden3])
            W_3, b_3 = W.eval(sess), b.eval(sess)
            #np.savetxt("weights/hidden3_W.csv", W_val, delimiter=",")
            #np.savetxt("weights/hidden3_b.csv", b_val, delimiter=",")
        cPickle.dump( [W_1, b_1, W_2, b_2, W_3, b_3], open( "pretrain.pkl", "wb" ) )

def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
