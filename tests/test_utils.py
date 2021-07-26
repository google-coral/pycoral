# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test utils for benchmark and manual tests."""

import argparse
import collections
import contextlib
import os
import random
import sys
import tempfile

import numpy as np
from PIL import Image
import unittest

_TEST_DATA_DIR = ''


def get_test_tmpdir():
  """Returns default test temp dir."""
  tmpdir = os.environ.get('TMPDIR', '')
  if not tmpdir:
    tmpdir = os.path.join(tempfile.gettempdir(), 'coral')
  if not os.path.exists(tmpdir):
    os.mkdir(tmpdir)
  return tmpdir


def test_data_path(path, *paths):
  """Returns absolute path for a given test file."""
  return os.path.abspath(os.path.join(_TEST_DATA_DIR, path, *paths))


@contextlib.contextmanager
def test_image(path, *paths):
  """Returns opened test image."""
  with open(test_data_path(path, *paths), 'rb') as f:
    with Image.open(f) as image:
      yield image


@contextlib.contextmanager
def temporary_file(suffix=None):
  """Creates a named temp file, and deletes after going out of scope.

  Exists to work around issues with passing the result of
  tempfile.NamedTemporaryFile to native code on Windows,
  if delete was set to True.

  Args:
    suffix: If provided, the file name will end with suffix.

  Yields:
    An file-like object.
  """
  resource = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
  try:
    yield resource
  finally:
    resource.close()
    os.unlink(resource.name)


def generate_random_input(seed, n):
  """Generates a list with n uint8 numbers."""
  random.seed(a=seed)
  return [random.randint(0, 255) for _ in range(n)]


def prepare_images(image_list, directory, shape):
  """Reads images and converts them to numpy array with specified shape.

  Args:
    image_list: a list of strings storing file names.
    directory: string, path of directory storing input images.
    shape: a 2-D tuple represents the shape of required input tensor.

  Returns:
    A list of numpy.array.
  """
  ret = []
  for filename in image_list:
    with open(os.path.join(directory, filename), 'rb') as f:
      with Image.open(f) as img:
        img = img.resize(shape, Image.NEAREST)
        ret.append(np.asarray(img).flatten())
  return np.array(ret)


def area(box):
  """Calculates area of a given bounding box."""
  assert box[1][0] >= box[0][0]
  assert box[1][1] >= box[0][1]
  return float((box[1][0] - box[0][0]) * (box[1][1] - box[0][1]))


def iou(box_a, box_b):
  """Calculates intersection area / union area for two bounding boxes."""
  assert area(box_a) > 0
  assert area(box_b) > 0
  intersect = np.array(
      [[max(box_a[0][0], box_b[0][0]),
        max(box_a[0][1], box_b[0][1])],
       [min(box_a[1][0], box_b[1][0]),
        min(box_a[1][1], box_b[1][1])]])
  return area(intersect) / (area(box_a) + area(box_b) - area(intersect))


def prepare_data_set_from_directory(path, test_ratio, fixed_order):
  """Parses data set from given directory, split them into train/test sets.

  Args:
    path: string, path of the data set. Images are stored in sub-directory named
      by category.
    test_ratio: float in (0,1), ratio of data used for testing.
    fixed_order: bool, whether to spilt data set in fixed order.

  Returns:
    (train_set, test_set), A tuple of two OrderedDicts. Keys are the categories
    and values are lists of image file names.
  """
  train_set = collections.OrderedDict()
  test_set = collections.OrderedDict()
  sub_dirs = os.listdir(path)
  if fixed_order:
    sub_dirs.sort()
  for category in sub_dirs:
    category_dir = os.path.join(path, category)
    if os.path.isdir(category_dir):
      images = [
          f for f in os.listdir(category_dir)
          if os.path.isfile(os.path.join(category_dir, f))
      ]
      if images:
        if fixed_order:
          images.sort()
        k = int(test_ratio * len(images))
        test_set[category] = images[:k]
        assert test_set[category], 'No images to test [{}]'.format(category)
        train_set[category] = images[k:]
        assert train_set[category], 'No images to train [{}]'.format(category)
  return train_set, test_set


def coral_test_main():
  """Test main to get test_data_dir flag from commend line.

  In edgetpu GoB repo:
    the Python test files are under edgetpu/tests.
    test_data is under edgetpu/test_data.
  """

  print('Python version: ', sys.version)
  global _TEST_DATA_DIR
  test_data_dir_default = os.path.join(
      os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'test_data')
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--test_data_dir', default=test_data_dir_default, type=str)
  args, sys.argv = parser.parse_known_args(sys.argv)
  _TEST_DATA_DIR = args.test_data_dir
  unittest.main()
