#!/bin/bash
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

BUILD_DIR="_build"
PREVIEW_DIR="${BUILD_DIR}/preview"
WEB_DIR="${BUILD_DIR}/web"

makeAll() {
  makeClean
  makeSphinxPreview
  makeSphinxWeb
}

makeSphinxWeb() {
  echo "Building Sphinx files for website..."
  sphinx-build -b html . ${WEB_DIR}
  # Delete intermediary/unused files:
  find ${WEB_DIR} -mindepth 1 -not -name "*.md" -delete
  rm ${WEB_DIR}/search.md ${WEB_DIR}/genindex.md ${WEB_DIR}/py-modindex.md
  # Some custom tweaks to the output:
  python3 postprocess.py -f ${WEB_DIR}/
  echo "All done. Web pages are in ${WEB_DIR}."
}

makeSphinxPreview() {
  echo "Building Sphinx files for local preview..."
  # Build the docs for local viewing (in "read the docs" style):
  sphinx-build -b html . ${PREVIEW_DIR} \
    -D html_theme="sphinx_rtd_theme" \
    -D html_file_suffix=".html" \
    -D html_link_suffix=".html"
  echo "All done. Preview pages are in ${PREVIEW_DIR}."
}

makeClean() {
  rm -rf ${BUILD_DIR}
  echo "Deleted ${BUILD_DIR}."
}

usage() {
  echo -n "Usage:
 makedocs.sh [-a|-w|-p|-c]

 Options (only one allowed):
  -a   Clean and make all docs (default)
  -w   Make Sphinx for website
  -p   Make Sphinx for local preview
  -c   Clean
"
}

if [[ "$#" -gt 1 ]]; then
  usage
elif [[ "$#" -eq 1 ]]; then
  if [[ "$1" = "-a" ]]; then
    makeAll
  elif [[ "$1" = "-w" ]]; then
    makeSphinxWeb
  elif [[ "$1" = "-p" ]]; then
    makeSphinxPreview
  elif [[ "$1" = "-c" ]]; then
    makeClean
  else
    usage
  fi
else
    makeAll
fi
