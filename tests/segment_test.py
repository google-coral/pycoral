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
from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter
from tests.test_utils import coral_test_main
from tests.test_utils import test_data_path


def deeplab_model_dm05(tpu):
  suffix = '_edgetpu' if tpu else ''
  return 'deeplabv3_mnv2_dm05_pascal_quant%s.tflite' % suffix


def deeplab_model_dm10(tpu):
  suffix = '_edgetpu' if tpu else ''
  return 'deeplabv3_mnv2_pascal_quant%s.tflite' % suffix


def keras_post_training_unet_mv2(tpu, size):
  suffix = '_edgetpu' if tpu else ''
  return 'keras_post_training_unet_mv2_%d_quant%s.tflite' % (size, suffix)


def array_iou(a, b):
  count = (a == b).sum()
  return count / (a.size + b.size - count)


def segment_image(model_file, image_file, mask_file):
  interpreter = make_interpreter(test_data_path(model_file))
  interpreter.allocate_tensors()

  image = Image.open(test_data_path(image_file)).resize(
      common.input_size(interpreter), Image.ANTIALIAS)
  common.set_input(interpreter, image)
  interpreter.invoke()

  result = segment.get_output(interpreter)
  if len(result.shape) > 2:
    result = np.argmax(result, axis=2)

  reference = np.asarray(Image.open(test_data_path(mask_file)))
  return array_iou(result, reference)


class SegmentTest(unittest.TestCase):

  def test_deeplab_dm10(self):
    for tpu in [False, True]:
      with self.subTest(tpu=tpu):
        self.assertGreater(
            segment_image(
                deeplab_model_dm10(tpu), 'bird_segmentation.bmp',
                'bird_segmentation_mask.bmp'), 0.90)

  def test_deeplab_dm05(self):
    for tpu in [False, True]:
      with self.subTest(tpu=tpu):
        self.assertGreater(
            segment_image(
                deeplab_model_dm05(tpu), 'bird_segmentation.bmp',
                'bird_segmentation_mask.bmp'), 0.90)

  def test_keras_post_training_unet_mv2_128(self):
    for tpu in [False, True]:
      with self.subTest(tpu=tpu):
        self.assertGreater(
            segment_image(
                keras_post_training_unet_mv2(tpu, 128), 'dog_segmentation.bmp',
                'dog_segmentation_mask.bmp'), 0.86)

  def test_keras_post_training_unet_mv2_256(self):
    for tpu in [False, True]:
      with self.subTest(tpu=tpu):
        self.assertGreater(
            segment_image(
                keras_post_training_unet_mv2(tpu, 256),
                'dog_segmentation_256.bmp', 'dog_segmentation_mask_256.bmp'),
            0.81)


if __name__ == '__main__':
  coral_test_main()
