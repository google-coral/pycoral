#!/bin/bash
#
# Download models, labels, and inputs for example code
#
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

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly TEST_DATA_URL="https://github.com/google-coral/test_data/raw/master/"
readonly TEST_DATA_DIR="${SCRIPT_DIR}/../test_data"

download() {
  for file in $@; do
    # Verify the file exists
    response="$(curl -Lso /dev/null -w "%{http_code}" \
      "${TEST_DATA_URL}/${file}")"
    if [[ "${response}" = "200" ]]; then
      echo "DOWNLOAD: ${file}"
      # Handle subdirectories in the file path (such as pipeline files)
      if [[ "${file}" == *"/"* ]]; then
        subdir="$(dirname "${file}")"
        mkdir -p "${TEST_DATA_DIR}/${subdir}"
        (cd "${TEST_DATA_DIR}/${subdir}" && curl -OL "${TEST_DATA_URL}/${file}")
      else
        (cd "${TEST_DATA_DIR}" && curl -OL "${TEST_DATA_URL}/${file}")
      fi
    else
      echo "NOT FOUND: ${file}"
    fi
  done
}

function get_backprop() {
  download \
    "mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite" \
    "sunflower.bmp"
}

function get_classification() {
  download \
    "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite" \
    "mobilenet_v2_1.0_224_inat_bird_quant.tflite" \
    "inat_bird_labels.txt" \
    "parrot.jpg"
}

function get_detection() {
  download \
    "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite" \
    "coco_labels.txt" \
    "grace_hopper.bmp"
}

function get_imprinting() {
  download \
    "mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite" \
    "cat.bmp"
}

function get_pipelining() {
  download \
    "pipeline/inception_v3_299_quant_segment_0_of_2_edgetpu.tflite" \
    "pipeline/inception_v3_299_quant_segment_1_of_2_edgetpu.tflite" \
    "imagenet_labels.txt" \
    "parrot.jpg"
}

function get_segmentation() {
  download \
    "deeplabv3_mnv2_pascal_quant_edgetpu.tflite" \
    "bird.bmp"
}

function get_small_detection() {
  download \
    "ssd_mobilenet_v2_coco_quant_no_nms_edgetpu.tflite" \
    "coco_labels.txt" \
    "kite_and_cold.jpg"
}

function get_two_models() {
  download \
    "mobilenet_v2_1.0_224_quant_edgetpu.tflite" \
    "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite" \
    "parrot.jpg"
}

function get_movenet() {
  download \
    "movenet_single_pose_lightning_ptq_edgetpu.tflite" \
    "squat.bmp"
}
function get_all() {
  echo "Download all files..."
  get_backprop
  get_classification
  get_detection
  get_imprinting
  get_pipelining
  get_segmentation
  get_small_detection
  get_two_models
  get_movenet
}

function usage() {
  echo -n "Usage:
  install_requirements.sh [filename]

  Provide the name of an example file in this directory to
  download only the files required for that example.
  If no filename provided, it downloads all example files.
  All files go into the pycoral/test_data/ directory."
}

function main() {
  if [[ "$#" -gt 1 ]]; then
    usage
    exit
  elif [[ "$#" -eq 0 ]]; then
    get_all
    exit
  fi

  case "$1" in
    backprop_last_layer.py)
      get_backprop
      ;;
    classify_image.py)
      get_classification
      ;;
    detect_image.py)
      get_detection
      ;;
    imprinting_learning.py)
      get_imprinting
      ;;
    model_pipelining_classify_image.py)
      get_pipelining
      ;;
    semantic_segmentation.py)
      get_segmentation
      ;;
    small_object_detection.py)
      get_small_detection
      ;;
    two_models_inference.py)
      get_two_models
      ;;
    movenet_pose_estimation.py)
      get_movenet
      ;;
    *)
      usage
      ;;
  esac
}

main "$@"
