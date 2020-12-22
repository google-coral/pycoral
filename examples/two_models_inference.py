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
r"""Demo to show running two models on one/two Edge TPU devices.

This is a dummy example that compares running two different models using one
Edge TPU vs two Edge TPUs. It requires that your system includes two Edge TPU
devices.

You give the script one classification model and one
detection model, and it runs each model the number of times specified with the
`num_inferences` argument, using the same image. It then reports the time
spent using either one or two Edge TPU devices.

For example:
```
bash examples/install_requirements.sh two_models_inference.py

python3 examples/two_models_inference.py \
  --classification_model test_data/mobilenet_v2_1.0_224_quant_edgetpu.tflite  \
  --detection_model \
    test_data/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite \
  --image test_data/parrot.jpg
```

Note: Running two models alternatively with one Edge TPU is cache unfriendly,
as each model continuously kicks the other model off the device's cache when
they each run. In this case, running several inferences with one model in a
batch before switching to another model can help to some extent. It's also
possible to co-compile both models so they can be cached simultaneously
(if they fit; read more at coral.ai/docs/edgetpu/compiler/). But using two
Edge TPUs with two threads can help more.
"""

import argparse
import contextlib
import threading
import time
from PIL import Image

from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import list_edge_tpus
from pycoral.utils.edgetpu import make_interpreter


@contextlib.contextmanager
def open_image(path):
  with open(path, 'rb') as f:
    with Image.open(f) as image:
      yield image


def run_two_models_one_tpu(classification_model, detection_model, image_name,
                           num_inferences, batch_size):
  """Runs two models ALTERNATIVELY using one Edge TPU.

  It runs classification model `batch_size` times and then switch to run
  detection model `batch_size` time until each model is run `num_inferences`
  times.

  Args:
    classification_model: string, path to classification model
    detection_model: string, path to detection model.
    image_name: string, path to input image.
    num_inferences: int, number of inferences to run for each model.
    batch_size: int, indicates how many inferences to run one model before
      switching to the other one.

  Returns:
    double, wall time it takes to finish the job.
  """
  start_time = time.perf_counter()
  interpreter_a = make_interpreter(classification_model, device=':0')
  interpreter_a.allocate_tensors()
  interpreter_b = make_interpreter(detection_model, device=':0')
  interpreter_b.allocate_tensors()

  with open_image(image_name) as image:
    size_a = common.input_size(interpreter_a)
    common.set_input(interpreter_a, image.resize(size_a, Image.NEAREST))
    _, scale_b = common.set_resized_input(
        interpreter_b, image.size,
        lambda size: image.resize(size, Image.NEAREST))

  num_iterations = (num_inferences + batch_size - 1) // batch_size
  for _ in range(num_iterations):
    for _ in range(batch_size):
      interpreter_a.invoke()
      classify.get_classes(interpreter_a, top_k=1)
    for _ in range(batch_size):
      interpreter_b.invoke()
      detect.get_objects(interpreter_b, score_threshold=0., image_scale=scale_b)
  return time.perf_counter() - start_time


def run_two_models_two_tpus(classification_model, detection_model, image_name,
                            num_inferences):
  """Runs two models using two Edge TPUs with two threads.

  Args:
    classification_model: string, path to classification model
    detection_model: string, path to detection model.
    image_name: string, path to input image.
    num_inferences: int, number of inferences to run for each model.

  Returns:
    double, wall time it takes to finish the job.
  """

  def classification_job(classification_model, image_name, num_inferences):
    """Runs classification job."""
    interpreter = make_interpreter(classification_model, device=':0')
    interpreter.allocate_tensors()
    size = common.input_size(interpreter)
    with open_image(image_name) as image:
      common.set_input(interpreter, image.resize(size, Image.NEAREST))

    for _ in range(num_inferences):
      interpreter.invoke()
      classify.get_classes(interpreter, top_k=1)

  def detection_job(detection_model, image_name, num_inferences):
    """Runs detection job."""
    interpreter = make_interpreter(detection_model, device=':1')
    interpreter.allocate_tensors()
    with open_image(image_name) as image:
      _, scale = common.set_resized_input(
          interpreter, image.size,
          lambda size: image.resize(size, Image.NEAREST))

    for _ in range(num_inferences):
      interpreter.invoke()
      detect.get_objects(interpreter, score_threshold=0., image_scale=scale)

  start_time = time.perf_counter()
  classification_thread = threading.Thread(
      target=classification_job,
      args=(classification_model, image_name, num_inferences))
  detection_thread = threading.Thread(
      target=detection_job, args=(detection_model, image_name, num_inferences))

  classification_thread.start()
  detection_thread.start()
  classification_thread.join()
  detection_thread.join()
  return time.perf_counter() - start_time


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--classification_model',
      help='Path of classification model.',
      required=True)
  parser.add_argument(
      '--detection_model', help='Path of detection model.', required=True)
  parser.add_argument('--image', help='Path of the image.', required=True)
  parser.add_argument(
      '--num_inferences',
      help='Number of inferences to run.',
      type=int,
      default=2000)
  parser.add_argument(
      '--batch_size',
      help='Runs one model batch_size times before switching to the other.',
      type=int,
      default=10)

  args = parser.parse_args()

  if len(list_edge_tpus()) <= 1:
    raise RuntimeError('This demo requires at least two Edge TPU available.')

  print('Running %s and %s with one Edge TPU, # inferences %d, batch_size %d.' %
        (args.classification_model, args.detection_model, args.num_inferences,
         args.batch_size))
  cost_one_tpu = run_two_models_one_tpu(args.classification_model,
                                        args.detection_model, args.image,
                                        args.num_inferences, args.batch_size)
  print('Running %s and %s with two Edge TPUs, # inferences %d.' %
        (args.classification_model, args.detection_model, args.num_inferences))
  cost_two_tpus = run_two_models_two_tpus(args.classification_model,
                                          args.detection_model, args.image,
                                          args.num_inferences)

  print('Inference with one Edge TPU costs %.2f seconds.' % cost_one_tpu)
  print('Inference with two Edge TPUs costs %.2f seconds.' % cost_two_tpus)


if __name__ == '__main__':
  main()
