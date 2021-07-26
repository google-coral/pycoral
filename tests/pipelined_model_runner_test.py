# Lint as: python3
# Copyright 2020 Google LLC
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
import time

import numpy as np

import pycoral.pipeline.pipelined_model_runner as pipeline
from pycoral.utils.edgetpu import list_edge_tpus
from pycoral.utils.edgetpu import make_interpreter
from tests import test_utils
import unittest


def _get_ref_result(ref_model, input_tensors):
  interpreter = make_interpreter(test_utils.test_data_path(ref_model))
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  assert len(input_details) == 1
  output_details = interpreter.get_output_details()
  assert len(output_details) == 1

  input_tensor, = input_tensors.values()
  interpreter.tensor(input_details[0]['index'])()[0][:, :] = input_tensor
  interpreter.invoke()
  return {
      output_details[0]['name']:
          np.array(interpreter.tensor(output_details[0]['index'])())
  }


def _get_devices(num_devices):
  """Returns list of device names in usb:N or pci:N format.

  This function prefers returning PCI Edge TPU first.

  Args:
    num_devices: int, number of devices expected

  Returns:
    list of devices in pci:N and/or usb:N format

  Raises:
    RuntimeError: if not enough devices are available
  """
  edge_tpus = list_edge_tpus()

  if len(edge_tpus) < num_devices:
    raise RuntimeError(
        'Not enough Edge TPUs detected, expected %d, detected %d.' %
        (num_devices, len(edge_tpus)))

  num_pci_devices = sum(1 for device in edge_tpus if device['type'] == 'pci')

  return ['pci:%d' % i for i in range(min(num_devices, num_pci_devices))] + [
      'usb:%d' % i for i in range(max(0, num_devices - num_pci_devices))
  ]


def _make_runner(model_paths, devices, allocate_tensors=True):
  print('Using devices: ', devices)
  print('Using models: ', model_paths)

  if len(model_paths) != len(devices):
    raise ValueError('# of devices and # of model_paths should match')

  interpreters = [
      make_interpreter(test_utils.test_data_path(m), d)
      for m, d in zip(model_paths, devices)
  ]
  if allocate_tensors:
    for interpreter in interpreters:
      interpreter.allocate_tensors()
  return pipeline.PipelinedModelRunner(interpreters)


class PipelinedModelRunnerTest(unittest.TestCase):

  _MODEL_SEGMENTS = [
      'pipeline/inception_v3_299_quant_segment_0_of_2_edgetpu.tflite',
      'pipeline/inception_v3_299_quant_segment_1_of_2_edgetpu.tflite',
  ]
  _REF_MODEL = 'inception_v3_299_quant_edgetpu.tflite'

  def _prepare_pipeline_inference(self,
                                  model_segments,
                                  ref_model=None,
                                  allocate_tensors=True):
    self.runner = _make_runner(model_segments,
                               _get_devices(len(model_segments)),
                               allocate_tensors)

    input_details = self.runner.interpreters()[0].get_input_details()
    self.assertEqual(len(input_details), 1)
    self.input_shape = input_details[0]['shape']

    np.random.seed(0)
    self.input_tensors = {
        'input':
            np.random.randint(0, 256, size=self.input_shape, dtype=np.uint8)
    }

    if ref_model:
      self.ref_result = _get_ref_result(ref_model, self.input_tensors)
    else:
      self.ref_result = None

  def test_bad_segments(self):
    bad_model_segments = [
        'pipeline/inception_v3_299_quant_segment_1_of_2_edgetpu.tflite',
        'pipeline/inception_v3_299_quant_segment_0_of_2_edgetpu.tflite',
    ]
    with self.assertRaisesRegex(
        ValueError, r'Interpreter [\d]+ can not get its input tensors'):
      unused_runner = _make_runner(bad_model_segments,
                                   [None] * len(self._MODEL_SEGMENTS))

  def test_unsupported_input_type(self):
    self._prepare_pipeline_inference(self._MODEL_SEGMENTS)
    with self.assertRaisesRegex(
        ValueError, 'Input should be a list of numpy array of type*'):
      self.runner.push({'input': np.random.random(self.input_shape)})

  def test_check_unconsumed_tensor(self):
    # Everything should work fine without crashing.
    self._prepare_pipeline_inference(self._MODEL_SEGMENTS)
    self.runner.push(self.input_tensors)

  def test_push_and_pop(self):
    self._prepare_pipeline_inference(self._MODEL_SEGMENTS, self._REF_MODEL)
    self.runner.push(self.input_tensors)
    result = self.runner.pop()
    np.testing.assert_equal(result, self.ref_result)

    # Check after {} is pushed.
    self.runner.push({})
    with self.assertRaisesRegex(RuntimeError,
                                'Pipeline was turned off before.'):
      self.runner.push(self.input_tensors)
    self.assertIsNone(self.runner.pop())

  def test_producer_and_consumer_threads(self):
    self._prepare_pipeline_inference(self._MODEL_SEGMENTS, self._REF_MODEL)
    num_requests = 5

    def producer(self):
      for _ in range(num_requests):
        self.runner.push(self.input_tensors)
      self.runner.push({})

    def consumer(self):
      while True:
        result = self.runner.pop()
        if not result:
          break
        np.testing.assert_equal(result, self.ref_result)

    producer_thread = threading.Thread(target=producer, args=(self,))
    consumer_thread = threading.Thread(target=consumer, args=(self,))

    producer_thread.start()
    consumer_thread.start()
    producer_thread.join()
    consumer_thread.join()

  def test_set_input_and_output_queue_size(self):
    self._prepare_pipeline_inference(self._MODEL_SEGMENTS)
    self.runner.set_input_queue_size(1)
    self.runner.set_output_queue_size(1)
    num_segments = len(self.runner.interpreters())

    # When both input and output queue size are set to 1, the max number of
    # requests pipeline runner can buffer is 2*num_segments+1. This is because
    # the intermediate queues need to be filled as well.
    max_buffered_requests = 2 * num_segments + 1

    # Push `max_buffered_requests` to pipeline, such that the next `push` will
    # be blocking as there is no consumer to process the results at the moment.
    for _ in range(max_buffered_requests):
      self.runner.push(self.input_tensors)

    # Sleep for `max_buffered_requests` seconds to make sure the first request
    # already reaches the last segments. This assumes that it takes 1 second for
    # each segment to return inference result (which is a generous upper bound).
    time.sleep(max_buffered_requests)

    def push_new_request(self):
      self.runner.push(self.input_tensors)
      self.runner.push({})

    producer_thread = threading.Thread(target=push_new_request, args=(self,))
    producer_thread.start()
    # If runner's input queue has room, push is non-blocking and should return
    # immediately. If producer_thread is still alive after join() with some
    # `timeout`, it means the thread is blocked.
    producer_thread.join(1.0)
    self.assertTrue(producer_thread.is_alive())

    processed_requests = 0
    while True:
      result = self.runner.pop()
      if not result:
        break
      processed_requests += 1
    self.assertEqual(processed_requests, max_buffered_requests + 1)
    producer_thread.join(1.0)
    self.assertFalse(producer_thread.is_alive())

  def test_interpreter_inference_error(self):
    self._prepare_pipeline_inference(
        self._MODEL_SEGMENTS, allocate_tensors=False)
    # Interpreter error can only be caught by the first pop call.
    self.runner.push(self.input_tensors)
    with self.assertRaisesRegex(
        RuntimeError,
        'Segment 0 runner error: Invoke called on model that is not ready.'):
      self.runner.pop()
    # Once error occurs, the following Push calls will fail.
    with self.assertRaisesRegex(
        RuntimeError,
        'Segment 0 runner error: Invoke called on model that is not ready.'):
      self.runner.push(self.input_tensors)


if __name__ == '__main__':
  test_utils.coral_test_main()
