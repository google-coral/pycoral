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
import numpy as np

from PIL import Image
import unittest
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils import edgetpu
from tests import test_utils

CHICKADEE = 20
TABBY_CAT = 282
TIGER_CAT = 283
EGYPTIAN_CAT = 286

EFFICIENTNET_IMAGE_QUANTIZATION = (1 / 128, 127)


def test_image(image_file, size):
  return Image.open(test_utils.test_data_path(image_file)).resize(
      size, Image.NEAREST)


def rescale_image(image, image_quantization, tensor_quatization, tensor_dtype):
  scale0, zero_point0 = image_quantization
  scale, zero_point = tensor_quatization

  min_value = np.iinfo(tensor_dtype).min
  max_value = np.iinfo(tensor_dtype).max

  def rescale(x):
    # The following is the same as y = (x - a) / b, where
    # b = scale / scale0 and a = zero_point0 - b * zero_point.
    y = int(zero_point + (scale0 * (x - zero_point0)) / scale)
    return max(min_value, min(y, max_value))

  rescale = np.vectorize(rescale, otypes=[tensor_dtype])
  return rescale(image)


def classify_image(model_file, delegate, image_file, image_quantization=None):
  """Runs image classification and returns result with the highest score.

  Args:
    model_file: string, model file name.
    delegate: Edge TPU delegate.
    image_file: string, image file name.
    image_quantization: (scale: float, zero_point: float), assumed image
      quantization parameters.

  Returns:
    Classification result with the highest score as (index, score) tuple.
  """
  interpreter = edgetpu.make_interpreter(
      test_utils.test_data_path(model_file), delegate=delegate)
  interpreter.allocate_tensors()
  image = test_image(image_file, common.input_size(interpreter))

  input_type = common.input_details(interpreter, 'dtype')
  if np.issubdtype(input_type, np.floating):
    # This preprocessing is specific to MobileNet V1 with floating point input.
    image = (input_type(image) - 127.5) / 127.5

  if np.issubdtype(input_type, np.integer) and image_quantization:
    image = rescale_image(image, image_quantization,
                          common.input_details(interpreter, 'quantization'),
                          input_type)

  common.set_input(interpreter, image)
  interpreter.invoke()
  return classify.get_classes(interpreter)[0]


def mobilenet_v1(depth_multiplier, input_size):
  return 'mobilenet_v1_%s_%d_quant_edgetpu.tflite' % (depth_multiplier,
                                                      input_size)


def mobilenet_v1_float_io(depth_multiplier, input_size):
  return 'mobilenet_v1_%s_%d_ptq_float_io_legacy_edgetpu.tflite' % (
      depth_multiplier, input_size)


def efficientnet(input_type):
  return 'efficientnet-edgetpu-%s_quant_edgetpu.tflite' % input_type


class TestClassify(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(TestClassify, cls).setUpClass()
    cls.delegate = edgetpu.load_edgetpu_delegate()

  def test_mobilenet_v1_100_224(self):
    index, score = classify_image(
        mobilenet_v1(1.0, 224), self.delegate, 'cat.bmp')
    self.assertEqual(index, EGYPTIAN_CAT)
    self.assertGreater(score, 0.78)

  def test_mobilenet_v1_050_160(self):
    index, score = classify_image(
        mobilenet_v1(0.5, 160), self.delegate, 'cat.bmp')
    self.assertEqual(index, EGYPTIAN_CAT)
    self.assertGreater(score, 0.67)

  def test_mobilenet_v1_float_224(self):
    index, score = classify_image(
        mobilenet_v1_float_io(1.0, 224), self.delegate, 'cat.bmp')
    self.assertEqual(index, EGYPTIAN_CAT)
    self.assertGreater(score, 0.7)

  def test_efficientnet_l(self):
    index, score = classify_image(
        efficientnet('L'), self.delegate, 'cat.bmp',
        EFFICIENTNET_IMAGE_QUANTIZATION)
    self.assertEqual(index, EGYPTIAN_CAT)
    self.assertGreater(score, 0.45)


if __name__ == '__main__':
  test_utils.coral_test_main()
