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
"""Benchmarks imprinting inference time under online training mode on small data set."""

import collections
import sys
import time

import numpy as np

from benchmarks import benchmark_utils
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.learn.imprinting import engine
from pycoral.utils import edgetpu
import tflite_runtime.interpreter as tflite


def run_benchmark(model, delegate):
  """Measures training time for given model with random data.

  Args:
    model: string, file name of the input model.
    delegate: Edge TPU delegate.

  Returns:
    float, training time in ms.
  """
  imprinting_engine = engine.ImprintingEngine(
      benchmark_utils.test_data_path(model), keep_classes=False)

  extractor = edgetpu.make_interpreter(
      imprinting_engine.serialize_extractor_model(), delegate=delegate)
  extractor.allocate_tensors()
  width, height = common.input_size(extractor)

  np.random.seed(12345)

  # 10 Categories, each has 20 images.
  data_by_category = collections.defaultdict(list)
  for i in range(10):
    for _ in range(20):
      data_by_category[i].append(
          np.random.randint(0, 256, (height, width, 3), dtype=np.uint8))

  inference_time = 0.
  for class_id, tensors in enumerate(data_by_category.values()):
    for tensor in tensors:
      common.set_input(extractor, tensor)
      extractor.invoke()
      imprinting_engine.train(classify.get_scores(extractor), class_id=class_id)

    start = time.perf_counter()
    interpreter = tflite.Interpreter(
        model_content=imprinting_engine.serialize_model(),
        experimental_delegates=[delegate])
    interpreter.allocate_tensors()
    common.set_input(interpreter, tensors[0])
    interpreter.invoke()
    classify.get_classes(interpreter, top_k=3)
    inference_time += (time.perf_counter() - start) * 1000

  print('Model: %s' % model)
  print('Inference time: %.2fms' % inference_time)
  return inference_time


def main():
  print('Python version: ', sys.version)
  args = benchmark_utils.parse_args()
  machine = benchmark_utils.machine_info()
  benchmark_utils.check_cpu_scaling_governor_status()
  models, reference = benchmark_utils.read_reference(
      'imprinting_reference_inference_%s.csv' % machine)
  results = [('MODEL', 'DATA_SET', 'INFERENCE_TIME')]
  delegate = edgetpu.load_edgetpu_delegate()
  for i, name in enumerate(models, start=1):
    print('---------------- %d / %d ----------------' % (i, len(models)))
    results.append((name, 'random', run_benchmark(name, delegate)))
  benchmark_utils.save_as_csv(
      'imprinting_benchmarks_inference_%s_%s.csv' %
      (machine, time.strftime('%Y%m%d-%H%M%S')), results)
  benchmark_utils.check_result(reference, results, args.enable_assertion)


if __name__ == '__main__':
  main()
