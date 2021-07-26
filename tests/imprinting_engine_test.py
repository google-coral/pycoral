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

import collections
from PIL import Image

from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.learn.imprinting import engine
from pycoral.utils import edgetpu
from tests import test_utils
import unittest

_MODEL_LIST = [
    'mobilenet_v1_1.0_224_l2norm_quant.tflite',
    'mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite'
]

TrainPoint = collections.namedtuple('TainPoint', ['images', 'class_id'])
TestPoint = collections.namedtuple('TainPoint', ['image', 'class_id', 'score'])


def set_input(interpreter, image):
  size = common.input_size(interpreter)
  common.set_input(interpreter, image.resize(size, Image.NEAREST))


class TestImprintingEnginePythonAPI(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(TestImprintingEnginePythonAPI, cls).setUpClass()
    cls.delegate = edgetpu.load_edgetpu_delegate()

  def _train_and_test(self, model_path, train_points, test_points,
                      keep_classes):
    # Train.
    imprinting_engine = engine.ImprintingEngine(model_path, keep_classes)

    extractor = edgetpu.make_interpreter(
        imprinting_engine.serialize_extractor_model(), delegate=self.delegate)
    extractor.allocate_tensors()

    for point in train_points:
      for image in point.images:
        with test_utils.test_image('imprinting', image) as img:
          set_input(extractor, img)
          extractor.invoke()
          embedding = classify.get_scores(extractor)
          self.assertEqual(len(embedding), imprinting_engine.embedding_dim)
          imprinting_engine.train(embedding, point.class_id)

    # Test.
    trained_model = imprinting_engine.serialize_model()
    classifier = edgetpu.make_interpreter(trained_model, delegate=self.delegate)
    classifier.allocate_tensors()

    self.assertEqual(len(classifier.get_output_details()), 1)

    if not keep_classes:
      self.assertEqual(len(train_points), classify.num_classes(classifier))

    for point in test_points:
      with test_utils.test_image('imprinting', point.image) as img:
        set_input(classifier, img)
        classifier.invoke()
        top = classify.get_classes(classifier, top_k=1)[0]
        self.assertEqual(top.id, point.class_id)
        self.assertGreater(top.score, point.score)

    return trained_model

  # Test full model, not keeping base model classes.
  def test_training_l2_norm_model_not_keep_classes(self):
    train_points = [
        TrainPoint(images=['cat_train_0.bmp'], class_id=0),
        TrainPoint(images=['dog_train_0.bmp'], class_id=1),
        TrainPoint(
            images=['hotdog_train_0.bmp', 'hotdog_train_1.bmp'], class_id=2),
    ]
    test_points = [
        TestPoint(image='cat_test_0.bmp', class_id=0, score=0.99),
        TestPoint(image='dog_test_0.bmp', class_id=1, score=0.99),
        TestPoint(image='hotdog_test_0.bmp', class_id=2, score=0.99)
    ]
    for model_path in _MODEL_LIST:
      with self.subTest(model_path=model_path):
        self._train_and_test(
            test_utils.test_data_path(model_path),
            train_points,
            test_points,
            keep_classes=False)

  # Test full model, keeping base model classes.
  def test_training_l2_norm_model_keep_classes(self):
    train_points = [
        TrainPoint(images=['cat_train_0.bmp'], class_id=1001),
        TrainPoint(images=['dog_train_0.bmp'], class_id=1002),
        TrainPoint(
            images=['hotdog_train_0.bmp', 'hotdog_train_1.bmp'], class_id=1003)
    ]
    test_points = [
        TestPoint(image='cat_test_0.bmp', class_id=1001, score=0.99),
        TestPoint(image='hotdog_test_0.bmp', class_id=1003, score=0.92)
    ]
    for model_path in _MODEL_LIST:
      with self.subTest(model_path=model_path):
        self._train_and_test(
            test_utils.test_data_path(model_path),
            train_points,
            test_points,
            keep_classes=True)

  def test_incremental_training(self):
    train_points = [TrainPoint(images=['cat_train_0.bmp'], class_id=0)]
    retrain_points = [
        TrainPoint(images=['dog_train_0.bmp'], class_id=1),
        TrainPoint(
            images=['hotdog_train_0.bmp', 'hotdog_train_1.bmp'], class_id=2)
    ]
    test_points = [
        TestPoint(image='cat_test_0.bmp', class_id=0, score=0.99),
        TestPoint(image='dog_test_0.bmp', class_id=1, score=0.99),
        TestPoint(image='hotdog_test_0.bmp', class_id=2, score=0.99)
    ]
    for model_path in _MODEL_LIST:
      with self.subTest(model_path=model_path):
        model = self._train_and_test(
            test_utils.test_data_path(model_path),
            train_points, [],
            keep_classes=False)

        with test_utils.temporary_file(suffix='.tflite') as new_model_file:
          new_model_file.write(model)
          # Retrain based on cat only model.
          self._train_and_test(
              new_model_file.name,
              retrain_points,
              test_points,
              keep_classes=True)

  def test_imprinting_engine_saving_without_training(self):
    model_list = [
        'mobilenet_v1_1.0_224_l2norm_quant.tflite',
        'mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite'
    ]
    for model in model_list:
      imprinting_engine = engine.ImprintingEngine(
          test_utils.test_data_path(model), keep_classes=False)
      with self.assertRaisesRegex(RuntimeError, 'Model is not trained.'):
        imprinting_engine.serialize_model()

  def test_imprinting_engine_invalid_model_path(self):
    with self.assertRaisesRegex(
        ValueError, 'Failed to open file: invalid_model_path.tflite'):
      engine.ImprintingEngine('invalid_model_path.tflite')

  def test_imprinting_engine_load_extractor_with_wrong_format(self):
    expected_message = ('Unsupported model architecture. Input model must have '
                        'an L2Norm layer.')
    with self.assertRaisesRegex(ValueError, expected_message):
      engine.ImprintingEngine(
          test_utils.test_data_path('mobilenet_v1_1.0_224_quant.tflite'))


if __name__ == '__main__':
  test_utils.coral_test_main()
