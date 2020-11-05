# PyCoral API docs

This directory holds the source files required to build the PyCoral API
reference with Sphinx.

You can build the reference docs as follows:

```
# We require Python3, so if that's not your default, first start a virtual environment:
python3 -m venv ~/.my_venvs/coraldocs
source ~/.my_venvs/coraldocs/bin/activate

# Navigate to the pycoral/docs/ directory and run these commands...

# Install the doc build dependencies:
pip install -r requirements.txt

# Build the docs:
bash makedocs.sh
```

The results are output in `_build/`. The `_build/preview/` files are for local
viewing--just open the `index.html` page. The `_build/web/` files are designed
for publishing on the Coral website.

For more information about the syntax in these RST files, see the
[reStructuredText documentation](http://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html).
