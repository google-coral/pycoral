#!/bin/bash
#
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
set -ex

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly MAKEFILE="${SCRIPT_DIR}/../Makefile"
readonly DOCKER_CPUS="${DOCKER_CPUS:=k8 aarch64 armv7a}"
PYTHON_VERSIONS="36 37 38 39"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      make -f "${MAKEFILE}" clean
      shift
      ;;
    --python_versions)
      PYTHON_VERSIONS=$2
      shift
      shift
      ;;
    *)
      shift
      ;;
  esac
done

function docker_image {
  case $1 in
    36) echo "ubuntu:18.04" ;;
    37) echo "debian:buster" ;;
    38) echo "ubuntu:20.04" ;;
    39) echo "debian:bullseye" ;;
    *) echo "Unsupported python version: $1" 1>&2; exit 1 ;;
  esac
}

for python_version in ${PYTHON_VERSIONS}; do
  make DOCKER_CPUS="${DOCKER_CPUS}" \
       DOCKER_IMAGE=$(docker_image "${python_version}") \
       DOCKER_TARGETS="pybind tflite wheel tflite-wheel" \
       -f "${MAKEFILE}" \
       docker-build
done
