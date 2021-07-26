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
"""Benchmark of models.

Benchmark are measured with CPU 'performance' mode. To enable it, you need to
install 'cpupower' and run:
sudo cpupower frequency-set --governor performance

The reference number is measured on:
  - 'x86_64': Intel Xeon E5-1650 v3(3.50GHz) + Edge TPU accelarator + USB 3.0
  - 'rp3b': Raspberry Pi 3 B (version1.2)+ Edge TPU accelarator + USB 2.0
  - 'rp3b+': Raspberry Pi 3 B+ (version1.3)+ Edge TPU accelarator + USB 2.0
  - 'aarch64': Edge TPU dev board.
"""

import sys
import time
import timeit

import numpy as np

from benchmarks import benchmark_utils
from pycoral.utils import edgetpu


def run_benchmark(model, delegate):
  """Returns average inference time in ms on specified model with random input."""

  print('Benchmark for [%s]' % model)
  print('model path = %s' % benchmark_utils.test_data_path(model))
  interpreter = edgetpu.make_interpreter(
      benchmark_utils.test_data_path(model), delegate=delegate)
  interpreter.allocate_tensors()
  iterations = 200 if 'edgetpu' in model else 20

  input_tensor = interpreter.tensor(interpreter.get_input_details()[0]['index'])
  np.random.seed(12345)
  input_tensor()[0] = np.random.randint(
      0, 256, size=input_tensor().shape[1:], dtype=np.uint8)
  result = 1000 * timeit.timeit(
      interpreter.invoke, number=iterations) / iterations

  print('%.2f ms (iterations = %d)' % (result, iterations))
  return result


def main():
  print('Python version: ', sys.version)
  args = benchmark_utils.parse_args()
  machine = benchmark_utils.machine_info()
  benchmark_utils.check_cpu_scaling_governor_status()
  models, reference = benchmark_utils.read_reference(
      'inference_reference_%s.csv' % machine)

  results = [('MODEL', 'INFERENCE_TIME')]
  delegate = edgetpu.load_edgetpu_delegate()
  for i, model in enumerate(models, start=1):
    print('-------------- Model %d / %d ---------------' % (i, len(models)))
    results.append((model, run_benchmark(model, delegate)))
  benchmark_utils.save_as_csv(
      'inference_benchmarks_%s_%s.csv' %
      (machine, time.strftime('%Y%m%d-%H%M%S')), results)
  benchmark_utils.check_result(reference, results, args.enable_assertion)


if __name__ == '__main__':
  main()
