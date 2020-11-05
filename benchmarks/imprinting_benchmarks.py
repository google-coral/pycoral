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
"""Benchmarks imprinting training time on small data set."""

import collections
import time

import numpy as np

from benchmarks import test_utils
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.learn.imprinting.engine import ImprintingEngine
from pycoral.utils.edgetpu import make_interpreter


def run_benchmark(model):
  """Measures training time for given model with random data.

  Args:
    model: string, file name of the input model.

  Returns:
    float, training time in ms.
  """

  engine = ImprintingEngine(
      test_utils.test_data_path(model), keep_classes=False)

  extractor = make_interpreter(engine.serialize_extractor_model())
  extractor.allocate_tensors()
  width, height = common.input_size(extractor)

  np.random.seed(12345)

  # 10 Categories, each has 20 images.
  data_by_category = collections.defaultdict(list)
  for i in range(10):
    for _ in range(20):
      data_by_category[i].append(
          np.random.randint(0, 256, (height, width, 3), dtype=np.uint8))

  start = time.perf_counter()

  for class_id, tensors in enumerate(data_by_category.values()):
    for tensor in tensors:
      common.set_input(extractor, tensor)
      extractor.invoke()
      engine.train(classify.get_scores(extractor), class_id=class_id)

  engine.serialize_model()

  training_time = (time.perf_counter() - start) * 1000

  print('Model: %s' % model)
  print('Training time: %.2fms' % training_time)
  return training_time


def main():
  args = test_utils.parse_args()
  machine = test_utils.machine_info()
  models, reference = test_utils.read_reference(
      'imprinting_reference_training_%s.csv' % machine)
  results = [('MODEL', 'DATA_SET', 'TRAINING_TIME')]
  for i, name in enumerate(models, start=1):
    print('---------------- %d / %d ----------------' % (i, len(models)))
    results.append((name, 'random', run_benchmark(name)))
  test_utils.save_as_csv(
      'imprinting_benchmarks_training_%s_%s.csv' %
      (machine, time.strftime('%Y%m%d-%H%M%S')), results)
  test_utils.check_result(reference, results, args.enable_assertion)


if __name__ == '__main__':
  main()
