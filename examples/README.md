# PyCoral API examples

This directory contains several examples that show how to use the
[PyCoral API](https://coral.ai/docs/edgetpu/api-intro/) to perform
inference or on-device transfer learning.

## Run the examples

1.  [Set up your Coral device](https://coral.ai/docs/setup/) (includes steps
to install this PyCoral library).

2.  Clone this repo onto the host device (onto your Coral board, or if using
a Coral accelerator, then onto the host system):

    ```
    git clone https://github.com/google-coral/pycoral
    ```

3.  Download the required model and other files required to run each sample,
using the `install_requirements.sh` script. Pass this script the filename you
want to run and it will download just the files for that example. Or exclude the
filename to download the files for all examples:

    ```
    cd pycoral

    bash examples/install_requirements.sh
    ```

4.  Then run the example command shown at the top of each `.py` file to run
the code (using the files you just downloaded). Some examples also require
additional downloads, which are specified in the code comments at the top of the
file.

For more pre-compiled models, see [coral.ai/models](https://coral.ai/models/).

For more information about building models and running inference on the Edge
TPU, see the [Coral documentation](https://coral.ai/docs/).
