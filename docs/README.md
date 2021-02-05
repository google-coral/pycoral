# PyCoral API docs

This directory holds the source files required to build the PyCoral API
reference with Sphinx.

You can build the reference docs as follows:

```
# To ensure consistent results, use a Python virtual environment:
python3 -m venv ~/.my_venvs/coraldocs
source ~/.my_venvs/coraldocs/bin/activate

# Navigate to the pycoral/ dir and build the pybind APIs:
cd pycoral
bash scripts/build.sh
# OR, build for only the python version and architecture you need. Eg:
# DOCKER_CPUS=k8 scripts/build.sh --python_versions 38

# Navigate to the pycoral/docs/ dir and install the doc dependencies:
cd docs
pip install -r requirements.txt

# Build the docs:
bash makedocs.sh
```

The results are output in `_build/`. The `_build/preview/` files are for local
viewing--just open the `index.html` page. The `_build/web/` files are designed
for publishing on the Coral website.

For more information about the syntax in these RST files, see the
[reStructuredText documentation](http://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html).
