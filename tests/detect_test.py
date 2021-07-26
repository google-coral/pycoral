# Lint as: python3
# pylint:disable=g-generic-assert
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
from PIL import Image

import unittest
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils import edgetpu
from tests import test_utils

BBox = detect.BBox

CAT = 16  # coco_labels.txt
ABYSSINIAN = 0  # pet_labels.txt


def get_objects(model_file, delegate, image_file, score_threshold=0.0):
  interpreter = edgetpu.make_interpreter(
      test_utils.test_data_path(model_file), delegate=delegate)
  interpreter.allocate_tensors()
  image = Image.open(test_utils.test_data_path(image_file))
  _, scale = common.set_resized_input(
      interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
  interpreter.invoke()
  return detect.get_objects(
      interpreter, score_threshold=score_threshold, image_scale=scale)


def face_model():
  return 'ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite'


def tf1_coco_model(version):
  return 'ssd_mobilenet_v%d_coco_quant_postprocess_edgetpu.tflite' % version


def tf2_coco_model(version):
  if version == 1:
    return 'tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_edgetpu.tflite'
  else:
    return 'tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite'


def fine_tuned_model():
  return 'ssd_mobilenet_v1_fine_tuned_pet_edgetpu.tflite'


class BBoxTest(unittest.TestCase):

  def test_basic(self):
    bbox = BBox(100, 110, 200, 210)
    self.assertEqual(bbox.xmin, 100)
    self.assertEqual(bbox.ymin, 110)
    self.assertEqual(bbox.xmax, 200)
    self.assertEqual(bbox.ymax, 210)

    self.assertTrue(bbox.valid)

    self.assertEqual(bbox.width, 100)
    self.assertEqual(bbox.height, 100)

    self.assertEqual(bbox.area, 10000)

  def test_scale(self):
    self.assertEqual(BBox(1, 1, 10, 20).scale(3, 4), BBox(3, 4, 30, 80))

  def test_translate(self):
    self.assertEqual(BBox(1, 1, 10, 20).translate(10, 20), BBox(11, 21, 20, 40))

  def test_map(self):
    self.assertEqual(BBox(1.1, 2.1, 3.1, 4.1).map(int), BBox(1, 2, 3, 4))
    self.assertEqual(BBox(1.9, 2.9, 3.9, 4.9).map(int), BBox(1, 2, 3, 4))

  def test_intersect_valid(self):
    a = BBox(0, 0, 200, 200)
    b = BBox(100, 100, 300, 300)

    self.assertAlmostEqual(BBox.iou(a, b), 0.14286, delta=0.0001)
    self.assertEqual(BBox.intersect(a, b), BBox(100, 100, 200, 200))

  def test_intersect_invalid(self):
    a = BBox(0, 0, 10, 20)
    b = BBox(20, 30, 25, 35)
    self.assertAlmostEqual(BBox.iou(a, b), 0.0)
    self.assertEqual(BBox.intersect(a, b), BBox(20, 30, 10, 20))

  def test_union(self):
    self.assertEqual(
        BBox.union(BBox(0, 0, 10, 20), BBox(50, 50, 60, 70)),
        BBox(0, 0, 60, 70))


class DetectTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(DetectTest, cls).setUpClass()
    cls.delegate = edgetpu.load_edgetpu_delegate()

  def assert_bbox_almost_equal(self, first, second, overlap_factor=0.8):
    self.assertGreaterEqual(
        BBox.iou(first, second),
        overlap_factor,
        msg='iou(%s, %s) is less than expected' % (first, second))

  def test_face(self):
    objs = get_objects(face_model(), self.delegate, 'grace_hopper.bmp')
    self.assertEqual(len(objs), 1)
    self.assertGreater(objs[0].score, 0.996)
    self.assert_bbox_almost_equal(objs[0].bbox,
                                  BBox(xmin=125, ymin=40, xmax=402, ymax=363))

  def test_tf1_coco_v1(self):
    objs = get_objects(tf1_coco_model(version=1), self.delegate, 'cat.bmp')
    self.assertGreater(len(objs), 0)
    obj = objs[0]
    self.assertEqual(obj.id, CAT)
    self.assertGreater(obj.score, 0.7)
    self.assert_bbox_almost_equal(obj.bbox,
                                  BBox(xmin=29, ymin=39, xmax=377, ymax=347))

  def test_tf1_coco_v2(self):
    objs = get_objects(tf1_coco_model(version=2), self.delegate, 'cat.bmp')
    self.assertGreater(len(objs), 0)
    obj = objs[0]
    self.assertEqual(obj.id, CAT)
    self.assertGreater(obj.score, 0.9)
    self.assert_bbox_almost_equal(obj.bbox,
                                  BBox(xmin=43, ymin=35, xmax=358, ymax=333))

  def test_tf2_coco_v1(self):
    objs = get_objects(tf2_coco_model(version=1), self.delegate, 'cat.bmp')
    self.assertGreater(len(objs), 0)
    obj = objs[0]
    self.assertEqual(obj.id, CAT)
    self.assertGreater(obj.score, 0.7)
    self.assert_bbox_almost_equal(obj.bbox,
                                  BBox(xmin=43, ymin=35, xmax=358, ymax=333))

  def test_tf2_coco_v2(self):
    objs = get_objects(tf2_coco_model(version=2), self.delegate, 'cat.bmp')
    self.assertGreater(len(objs), 0)
    obj = objs[0]
    self.assertEqual(obj.id, CAT)
    self.assertGreater(obj.score, 0.7)
    self.assert_bbox_almost_equal(obj.bbox,
                                  BBox(xmin=43, ymin=35, xmax=358, ymax=333))

  def test_fine_tuned(self):
    objs = get_objects(fine_tuned_model(), self.delegate, 'cat.bmp')
    self.assertGreater(len(objs), 0)
    obj = objs[0]
    self.assertEqual(obj.id, ABYSSINIAN)
    self.assertGreater(obj.score, 0.88)
    self.assert_bbox_almost_equal(obj.bbox,
                                  BBox(xmin=177, ymin=37, xmax=344, ymax=216))


if __name__ == '__main__':
  test_utils.coral_test_main()
