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
import ctypes
import ctypes.util
import io

import numpy as np

from pycoral.utils import edgetpu
from tests import test_utils
import unittest


# Detect whether GStreamer is available.
# This code session is copied from utils/edgetpu.py.
class _GstMapInfo(ctypes.Structure):
  _fields_ = [
      ('memory', ctypes.c_void_p),  # GstMemory *memory
      ('flags', ctypes.c_int),  # GstMapFlags flags
      ('data', ctypes.c_void_p),  # guint8 *data
      ('size', ctypes.c_size_t),  # gsize size
      ('maxsize', ctypes.c_size_t),  # gsize maxsize
      ('user_data', ctypes.c_void_p * 4),  # gpointer user_data[4]
      ('_gst_reserved', ctypes.c_void_p * 4)
  ]  # GST_PADDING


_libgst = None
try:
  # pylint:disable=g-import-not-at-top
  import gi
  gi.require_version('Gst', '1.0')
  from gi.repository import Gst
  _libgst = ctypes.CDLL(ctypes.util.find_library('gstreamer-1.0'))
  _libgst.gst_buffer_map.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(_GstMapInfo), ctypes.c_int
  ]
  _libgst.gst_buffer_map.restype = ctypes.c_int
  _libgst.gst_buffer_unmap.argtypes = [
      ctypes.c_void_p, ctypes.POINTER(_GstMapInfo)
  ]
  _libgst.gst_buffer_unmap.restype = None
  Gst.init(None)
except (ImportError, ValueError, OSError):
  pass


def read_file(filename):
  with open(filename, mode='rb') as f:
    return f.read()


def required_input_array_size(interpreter):
  input_shape = interpreter.get_input_details()[0]['shape']
  return np.prod(input_shape)


# Use --config=asan for better coverage.
class TestEdgeTpuUtils(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(TestEdgeTpuUtils, cls).setUpClass()
    cls.delegate = edgetpu.load_edgetpu_delegate()

  def _default_test_model_path(self):
    return test_utils.test_data_path(
        'mobilenet_v1_1.0_224_quant_edgetpu.tflite')

  def test_load_from_model_file(self):
    edgetpu.make_interpreter(
        self._default_test_model_path(), delegate=self.delegate)

  def test_load_from_model_content(self):
    with io.open(self._default_test_model_path(), 'rb') as model_file:
      edgetpu.make_interpreter(model_file.read(), delegate=self.delegate)

  def test_load_from_invalid_model_path(self):
    with self.assertRaisesRegex(
        ValueError, 'Could not open \'invalid_model_path.tflite\'.'):
      edgetpu.make_interpreter('invalid_model_path.tflite')

  def test_load_with_device(self):
    edgetpu.make_interpreter(self._default_test_model_path(), device=':0')

  def test_load_with_nonexistent_device(self):
    # Assume that there can not be 1000 Edge TPU devices connected.
    with self.assertRaisesRegex(ValueError, 'Failed to load delegate'):
      edgetpu.make_interpreter(self._default_test_model_path(), device=':1000')

  def test_load_with_invalid_device_str(self):
    with self.assertRaisesRegex(ValueError, 'Failed to load delegate'):
      edgetpu.make_interpreter(self._default_test_model_path(), device='foo')

  def _run_inference_with_different_input_types(self, interpreter, input_data):
    """Tests inference with different input types.

    It doesn't check correctness of inference. Instead it checks inference
    repeatability with different input types.

    Args:
      interpreter : A tflite interpreter.
      input_data (list): A 1-D list as the input tensor.
    """
    output_index = interpreter.get_output_details()[0]['index']
    # numpy array
    np_input = np.asarray(input_data, np.uint8)
    edgetpu.run_inference(interpreter, np_input)
    ret = interpreter.tensor(output_index)()
    ret0 = np.copy(ret)
    # bytes
    bytes_input = bytes(input_data)
    edgetpu.run_inference(interpreter, bytes_input)
    ret = interpreter.tensor(output_index)()
    self.assertTrue(np.array_equal(ret0, ret))
    # ctypes
    edgetpu.run_inference(
        interpreter, (np_input.ctypes.data_as(ctypes.c_void_p), np_input.size))
    ret = interpreter.tensor(output_index)()
    self.assertTrue(np.array_equal(ret0, ret))
    # Gst buffer
    if _libgst:
      gst_input = Gst.Buffer.new_wrapped(bytes_input)
      edgetpu.run_inference(interpreter, gst_input)
      ret = interpreter.tensor(output_index)()
      self.assertTrue(np.array_equal(ret0, ret))
    else:
      print('Can not import gi. Skip test on Gst.Buffer input type.')

  def _run_inference_with_gst(self, interpreter, input_data):
    output_index = interpreter.get_output_details()[0]['index']
    bytes_input = bytes(input_data)
    gst_input = Gst.Buffer.new_wrapped(bytes_input)
    edgetpu.run_inference(interpreter, gst_input)
    ret = interpreter.tensor(output_index)()
    return np.copy(ret)

  def test_run_inference_with_different_types(self):
    interpreter = edgetpu.make_interpreter(
        self._default_test_model_path(), delegate=self.delegate)
    interpreter.allocate_tensors()
    input_size = required_input_array_size(interpreter)
    input_data = test_utils.generate_random_input(1, input_size)
    self._run_inference_with_different_input_types(interpreter, input_data)

  def test_run_inference_larger_input_size(self):
    interpreter = edgetpu.make_interpreter(
        self._default_test_model_path(), delegate=self.delegate)
    interpreter.allocate_tensors()
    input_size = required_input_array_size(interpreter)
    input_data = test_utils.generate_random_input(1, input_size + 1)
    self._run_inference_with_different_input_types(interpreter, input_data)

  def test_compare_expected_and_larger_input_size(self):
    if _libgst:
      interpreter = edgetpu.make_interpreter(
          self._default_test_model_path(), delegate=self.delegate)
      interpreter.allocate_tensors()
      input_size = required_input_array_size(interpreter)
      larger_input_data = test_utils.generate_random_input(1, input_size + 1)
      larger_ret = self._run_inference_with_gst(interpreter, larger_input_data)
      ret = self._run_inference_with_gst(interpreter,
                                         larger_input_data[:input_size])
      self.assertTrue(np.array_equal(ret, larger_ret))
    else:
      print('Can not import gi. Skip test on Gst.Buffer input type.')

  def test_run_inference_smaller_input_size(self):
    interpreter = edgetpu.make_interpreter(
        self._default_test_model_path(), delegate=self.delegate)
    interpreter.allocate_tensors()
    input_size = required_input_array_size(interpreter)
    input_data = test_utils.generate_random_input(1, input_size - 1)
    with self.assertRaisesRegex(ValueError,
                                'input size=150527, expected=150528'):
      self._run_inference_with_different_input_types(interpreter, input_data)

  def test_invoke_with_dma_buffer_model_not_ready(self):
    interpreter = edgetpu.make_interpreter(
        self._default_test_model_path(), delegate=self.delegate)
    input_size = 224 * 224 * 3
    # Note: Exception is triggered because interpreter.allocate_tensors() is not
    # called.
    with self.assertRaisesRegex(RuntimeError,
                                'Invoke called on model that is not ready.'):
      edgetpu.invoke_with_dmabuffer(interpreter._native_handle(), 0, input_size)

  def test_invoke_with_mem_buffer_model_not_ready(self):
    interpreter = edgetpu.make_interpreter(
        self._default_test_model_path(), delegate=self.delegate)
    input_size = 224 * 224 * 3
    np_input = np.zeros(input_size, dtype=np.uint8)
    # Note: Exception is triggered because interpreter.allocate_tensors() is not
    # called.
    with self.assertRaisesRegex(RuntimeError,
                                'Invoke called on model that is not ready.'):
      edgetpu.invoke_with_membuffer(interpreter._native_handle(),
                                    np_input.ctypes.data, input_size)

  def test_list_edge_tpu_paths(self):
    self.assertGreater(len(edgetpu.list_edge_tpus()), 0)

  def test_set_verbosity(self):
    # Simply sets the verbosity and ensure it returns success.
    self.assertTrue(edgetpu.set_verbosity(10))
    # Returns the verbosity back to zero.
    self.assertTrue(edgetpu.set_verbosity(0))


if __name__ == '__main__':
  test_utils.coral_test_main()
