# Lint as: python3
# Copyright 2021 Google LLC
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
r"""Example using PyCoral to estimate a single human pose with Edge TPU MoveNet.

To run this code, you must attach an Edge TPU to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

For more details about MoveNet and its best practices, please see
https://www.tensorflow.org/hub/tutorials/movenet

Example usage:
```
bash examples/install_requirements.sh movenet_pose_estimation.py

python3 examples/movenet_pose_estimation.py \
  --model test_data/movenet_single_pose_lightning_ptq_edgetpu.tflite  \
  --input test_data/squat.bmp
```
"""

import argparse

from PIL import Image
from PIL import ImageDraw
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

_NUM_KEYPOINTS = 17


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', required=True, help='File path of .tflite file.')
  parser.add_argument(
      '-i', '--input', required=True, help='Image to be classified.')
  parser.add_argument(
      '--output',
      default='movenet_result.jpg',
      help='File path of the output image.')
  args = parser.parse_args()

  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  img = Image.open(args.input)
  resized_img = img.resize(common.input_size(interpreter), Image.ANTIALIAS)
  common.set_input(interpreter, resized_img)

  interpreter.invoke()

  pose = common.output_tensor(interpreter, 0).copy().reshape(_NUM_KEYPOINTS, 3)
  print(pose)
  draw = ImageDraw.Draw(img)
  width, height = img.size
  for i in range(0, _NUM_KEYPOINTS):
    draw.ellipse(
        xy=[
            pose[i][1] * width - 2, pose[i][0] * height - 2,
            pose[i][1] * width + 2, pose[i][0] * height + 2
        ],
        fill=(255, 0, 0))
  img.save(args.output)
  print('Done. Results saved at', args.output)


if __name__ == '__main__':
  main()
