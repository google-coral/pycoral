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
"""Evaluates the accuracy of imprinting based transfer learning model."""

import contextlib
import os
from PIL import Image

from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.learn.imprinting import engine
from pycoral.utils import edgetpu
from tests import test_utils
import unittest


@contextlib.contextmanager
def test_image(path):
  with open(path, 'rb') as f:
    with Image.open(f) as image:
      yield image


class ImprintingEngineEvaluationTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(ImprintingEngineEvaluationTest, cls).setUpClass()
    cls.delegate = edgetpu.load_edgetpu_delegate()

  def _transfer_learn_and_evaluate(self, model_path, keep_classes, dataset_path,
                                   test_ratio, top_k_range):
    """Transfer-learns with given params and returns the evaluation result.

    Args:
      model_path: string, path of the base model.
      keep_classes: bool, whether to keep base model classes.
      dataset_path: string, path to the directory of dataset. The images should
        be put under sub-directory named by category.
      test_ratio: float, the ratio of images used for test.
      top_k_range: int, top_k range to be evaluated. The function will return
        accuracy from top 1 to top k.

    Returns:
      list of float numbers.
    """
    imprinting_engine = engine.ImprintingEngine(model_path, keep_classes)

    extractor = edgetpu.make_interpreter(
        imprinting_engine.serialize_extractor_model(), delegate=self.delegate)
    extractor.allocate_tensors()

    num_classes = imprinting_engine.num_classes

    print('---------------      Parsing dataset      ----------------')
    print('Dataset path:', dataset_path)

    # train in fixed order to ensure the same evaluation result.
    train_set, test_set = test_utils.prepare_data_set_from_directory(
        dataset_path, test_ratio, True)

    print('Image list successfully parsed! Number of Categories = ',
          len(train_set))
    print('---------------  Processing training data ----------------')
    print('This process may take more than 30 seconds.')
    train_input = []
    labels_map = {}
    for class_id, (category, image_list) in enumerate(train_set.items()):
      print('Processing {} ({} images)'.format(category, len(image_list)))
      train_input.append(
          [os.path.join(dataset_path, category, image) for image in image_list])
      labels_map[num_classes + class_id] = category

    # train
    print('----------------      Start training     -----------------')
    size = common.input_size(extractor)
    for class_id, images in enumerate(train_input):
      for image in images:
        with test_image(image) as img:
          common.set_input(extractor, img.resize(size, Image.NEAREST))
          extractor.invoke()
          imprinting_engine.train(
              classify.get_scores(extractor), class_id=num_classes + class_id)

    print('----------------     Training finished   -----------------')
    with test_utils.temporary_file(suffix='.tflite') as output_model_path:
      output_model_path.write(imprinting_engine.serialize_model())

      # Evaluate
      print('----------------     Start evaluating    -----------------')
      classifier = edgetpu.make_interpreter(
          output_model_path.name, delegate=self.delegate)
      classifier.allocate_tensors()

      # top[i] represents number of top (i+1) correct inference.
      top_k_correct_count = [0] * top_k_range
      image_num = 0
      for category, image_list in test_set.items():
        n = len(image_list)
        print('Evaluating {} ({} images)'.format(category, n))
        for image_name in image_list:
          with test_image(os.path.join(dataset_path, category,
                                       image_name)) as img:
            # Set threshold as a negative number to ensure we get top k
            # candidates even if its score is 0.
            size = common.input_size(classifier)
            common.set_input(classifier, img.resize(size, Image.NEAREST))
            classifier.invoke()
            candidates = classify.get_classes(classifier, top_k=top_k_range)

            for i in range(len(candidates)):
              candidate = candidates[i]
              if candidate.id in labels_map and \
                 labels_map[candidate.id] == category:
                top_k_correct_count[i] += 1
                break
        image_num += n
      for i in range(1, top_k_range):
        top_k_correct_count[i] += top_k_correct_count[i - 1]

    return [top_k_correct_count[i] / image_num for i in range(top_k_range)]

  def _test_oxford17_flowers_single(self, model_path, keep_classes, expected):
    top_k_range = len(expected)
    ret = self._transfer_learn_and_evaluate(
        test_utils.test_data_path(model_path), keep_classes,
        test_utils.test_data_path('oxford_17flowers'), 0.25, top_k_range)
    for i in range(top_k_range):
      self.assertGreaterEqual(ret[i], expected[i])

  # Evaluate with L2Norm full model, not keeping base model classes.
  def test_oxford17_flowers_l2_norm_model_not_keep_classes(self):
    self._test_oxford17_flowers_single(
        'mobilenet_v1_1.0_224_l2norm_quant.tflite',
        keep_classes=False,
        expected=[0.86, 0.94, 0.96, 0.97, 0.97])

  # Evaluate with L2Norm full model, keeping base model classes.
  def test_oxford17_flowers_l2_norm_model_keep_classes(self):
    self._test_oxford17_flowers_single(
        'mobilenet_v1_1.0_224_l2norm_quant.tflite',
        keep_classes=True,
        expected=[0.86, 0.94, 0.96, 0.96, 0.97])


if __name__ == '__main__':
  test_utils.coral_test_main()
