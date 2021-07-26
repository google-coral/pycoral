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
"""Setup script for pycoral library."""

import importlib.machinery
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
  path = os.path.join(os.path.abspath(os.path.dirname(__file__)), filename)
  with open(path, 'r', encoding='utf-8') as f:
    return f.read()


def find_version(text):
  match = re.search(r"^__version__\s*=\s*['\"](.*)['\"]\s*$", text,
                    re.MULTILINE)
  return match.group(1)


setup(
    name='pycoral',
    description='Coral Python API',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    license='Apache 2',
    version=find_version(read('pycoral/__init__.py')),
    author='Coral',
    author_email='coral-support@google.com',
    url='https://github.com/google-coral/pycoral',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=['tflite_runtime']),
    package_data={
        '': [
            os.environ.get('WRAPPER_NAME',
                           '*' + importlib.machinery.EXTENSION_SUFFIXES[-1])
        ]
    },
    install_requires=[
        'numpy>=1.16.0',
        'Pillow>=4.0.0',
        'tflite-runtime==2.5.0.post1',
    ],
    **({
        'has_ext_modules': lambda: True
    } if 'WRAPPER_NAME' in os.environ else {}),
    python_requires='>=3.5.2',
)
