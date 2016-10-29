
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.datasets import mnist

import cPickle
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.learn.python.learn.datasets import base

class DataSet(object):

  def __init__(self,
               images,
               labels,
               keras=False):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    assert images.shape[0] == labels.shape[0], (
      'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    self._num_examples = images.shape[0]
    if keras:
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    images = images.astype(numpy.float32)
    if keras:
      images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, data_dir='mnist.pkl', keras=False):
  if not keras:
    (train_images, train_labels), (validation_images, validation_labels), (test_images, test_labels) = cPickle.load(file(data_dir, 'rb'))
  else:
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    validation_labels = train_labels[-10000:]
    validation_images = train_images[-10000:]
    train_labels = train_labels[:-10000]
    train_images = train_images[:-10000]

  train = DataSet(train_images, train_labels, keras=keras)
  validation = DataSet(validation_images,
                       validation_labels, keras=keras)
  test = DataSet(test_images, test_labels, keras=keras)

  return base.Datasets(train=train, validation=validation, test=test)


def load_mnist(train_dir='MNIST-data'):
  return read_data_sets(train_dir)
