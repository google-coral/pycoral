# pycoral

This repository contains an easy-to-use Python API to work with Coral devices:

*   [Dev Board](https://coral.ai/products/dev-board/)
*   [USB Accelerator](https://coral.ai/products/accelerator/)

You can run inference and do transfer learning.

## Compilation

1.  Run `scripts/build.sh` to build pybind11-based native layer for different
    Linux architectures. Build is Docker-based, so you need to have it
    installed.

1.  Run `make wheel` to generate Python library wheel and then `pip3 install
    $(ls dist/*.whl)` to install it
