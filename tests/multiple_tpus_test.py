# Lint as: python3
# pylint:disable=# pylint:disable=g-generic-assert
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
import threading

from PIL import Image

from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from tests import test_utils
import unittest


class MultipleTpusTest(unittest.TestCase):

  def test_run_classification_and_detection(self):

    def classification_task(num_inferences):
      tid = threading.get_ident()
      print('Thread: %d, %d inferences for classification task' %
            (tid, num_inferences))
      labels = read_label_file(test_utils.test_data_path('imagenet_labels.txt'))
      model_name = 'mobilenet_v1_1.0_224_quant_edgetpu.tflite'
      interpreter = make_interpreter(
          test_utils.test_data_path(model_name), device=':0')
      interpreter.allocate_tensors()
      size = common.input_size(interpreter)
      print('Thread: %d, using device 0' % tid)
      with test_utils.test_image('cat.bmp') as img:
        for _ in range(num_inferences):
          common.set_input(interpreter, img.resize(size, Image.NEAREST))
          interpreter.invoke()
          ret = classify.get_classes(interpreter, top_k=1)
          self.assertEqual(len(ret), 1)
          self.assertEqual(labels[ret[0].id], 'Egyptian cat')
      print('Thread: %d, done classification task' % tid)

    def detection_task(num_inferences):
      tid = threading.get_ident()
      print('Thread: %d, %d inferences for detection task' %
            (tid, num_inferences))
      model_name = 'ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite'
      interpreter = make_interpreter(
          test_utils.test_data_path(model_name), device=':1')
      interpreter.allocate_tensors()
      print('Thread: %d, using device 1' % tid)
      with test_utils.test_image('cat.bmp') as img:
        for _ in range(num_inferences):
          _, scale = common.set_resized_input(
              interpreter,
              img.size,
              lambda size, image=img: image.resize(size, Image.ANTIALIAS))
          interpreter.invoke()
          ret = detect.get_objects(
              interpreter, score_threshold=0.7, image_scale=scale)
          self.assertEqual(len(ret), 1)
          self.assertEqual(ret[0].id, 16)  # cat
          expected_bbox = detect.BBox(
              xmin=int(0.1 * img.size[0]),
              ymin=int(0.1 * img.size[1]),
              xmax=int(0.7 * img.size[0]),
              ymax=int(1.0 * img.size[1]))
          self.assertGreaterEqual(
              detect.BBox.iou(expected_bbox, ret[0].bbox), 0.85)
      print('Thread: %d, done detection task' % tid)

    num_inferences = 2000
    t1 = threading.Thread(target=classification_task, args=(num_inferences,))
    t2 = threading.Thread(target=detection_task, args=(num_inferences,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()


if __name__ == '__main__':
  test_utils.coral_test_main()
