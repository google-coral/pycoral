# PyCoral API

This repository contains an easy-to-use Python API to run inferences and
perform on-device transfer learning with TensorFlow Lite models on
[Coral devices](https://coral.ai/products/).

You can install this library with the Python wheels listed at
[coral.ai/software/](https://coral.ai/software/#pycoral-api).

## Documentation and examples

To learn more about how to use the PyCoral API, see our guide to [Run inference
on the Edge TPU with Python](https://coral.ai/docs/edgetpu/tflite-python/) and
check out the [PyCoral API reference](https://coral.ai/docs/reference/py/).

Several Python examples are available in the `examples/` directory. For
instructions, see the [examples README](
https://github.com/google-coral/pycoral/tree/master/examples#pycoral-api-examples).


## Compilation

To build the library yourself, follow these steps:

1.  Clone this repo and include submodules:

    ```
    git clone --recurse-submodules https://github.com/google-coral/pycoral
    ```

    If you already cloned without the submodules. You can add them with this:

    ```
    cd pycoral

    git submodule init && git submodule update
    ```

1.  Run `scripts/build.sh` to build pybind11-based native layer for different
    Linux architectures. Build is Docker-based, so you need to have it
    installed.

1.  Run `make wheel` to generate Python library wheel and then `pip3 install
    $(ls dist/*.whl)` to install it
