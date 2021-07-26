# PyCoral changelog


## 2.0 (Grouper release)

* Improved error reporting for `PipelinedModelRunner`. It now prints error
messages originating from the TensorFlow Lite runtime.

**Changed APIs (code-breaking):**

* [`PipelinedModelRunner.push()`](https://coral.ai/docs/reference/py/pycoral.pipeline/#pycoral.pipeline.pipelined_model_runner.PipelinedModelRunner.push)
now requires a dictionary for the `input_tensors` (instead of a list), so that
each input tensor provides a corresponding tensor name as the dictionary key.
This method is also now void instead of returning a bool; it will raise
`RuntimeError` if the push fails.

* Similarly,
[`PipelinedModelRunner.pop()`](https://coral.ai/docs/reference/py/pycoral.pipeline/#pycoral.pipeline.pipelined_model_runner.PipelinedModelRunner.pop)
now returns a dictionary instead of a list, and also may raise `RuntimeError`.


**Updated APIs:**

* [`make_interpreter()`](https://coral.ai/docs/reference/py/pycoral.utils/#pycoral.utils.edgetpu.make_interpreter)
now accepts an optional `delegate` argument to specify the Edge TPU delegate
object you want to use.
* [`get_objects()`](https://coral.ai/docs/reference/py/pycoral.adapters/#pycoral.adapters.detect.get_objects)
now supports SSD models with different orders in the output tensor.


**New APIs:**

*  `utils.edgetpu.set_verbosity()` prints logs related to each Edge TPU.


## 1.0 (Frogfish release)

*   Initial pycoral release
