# PyCoral API docs

This directory holds the source files required to build the PyCoral API
reference with Sphinx.

You can build the reference docs as follows:

```
# Start in the pycoral repo root directory...
# First, be sure you sync the required submodules (needed to build pycoral):
git submodule init && git submodule update

# Build the "pywrap" lib for the python version and architecture you need. Eg:
DOCKER_CPUS=k8 scripts/build.sh --python_versions 39

# To ensure compatibility with GLIBC version used in the above library build,
# we should also build the docs in Docker. So open a matching docker shell
# using the same DOCKER_IMAGE that corresponds to the python_version used above
# (see pycoral/scripts/build.sh):
DOCKER_IMAGE=ubuntu:21.04 make docker-shell

# Inside the Docker shell, install libedgetpu:
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std

# Now go into the docs dir and install Python libs:
cd docs
sudo apt install python3-pip -y
python3 -m pip install -r requirements.txt
# Look for warnings about libraries installed in location outside your PATH.
# Add them like this:
export PATH=$PATH:/home/yourname/.local/bin

# Finally, build the docs:
bash makedocs.sh -p
```

The results are output in `_build/`. The `_build/preview/` files are for local
viewing--just open the `index.html` page. The `_build/web/` files are designed
for publishing on the Coral website (use the `-w` flag).

For more information about the syntax in these RST files, see the
[reStructuredText documentation](http://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html).
