# Lint as: python3
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
r"""An example to perform object detection with an image with added supports for smaller objects.

The following command runs this example for object detection using a
MobileNet model trained with the COCO dataset (it can detect 90 types
of objects):
```
bash examples/install_requirements.sh small_object_detection.py

python3 examples/small_object_detection.py \
  --model test_data/ssd_mobilenet_v2_coco_quant_no_nms_edgetpu.tflite \
  --label test_data/coco_labels.txt \
  --input test_data/kite_and_cold.jpg \
  --tile_size 1352x900,500x500,250x250 \
  --tile_overlap 50 \
  --score_threshold 0.5 \
  --output ${HOME}/object_detection_results.jpg
```

Note: this example demonstrate small object detection, using the method of
splitting the original image into tiles with some added overlaps in consecutive
tiles. The tile size can also be specified in multiple layers as
demonstrated on the above command. With the overlapping tiles and layers,
some object candidates may have overlapping bounding boxes. The example then
uses Non-Maximum-Suppressions to suppress the overlapping bounding boxes on the
same objects. It then saves the result of the given image at the location
specified by `output`, with bounding boxes drawn around each detected object.

In order to boost performance, the model has non_max_suppression striped from
the post processing operator. To do this, we can re-export the checkpoint by
setting the iou_threshold to 1. By doing so, we see an overall speedup of about
2x on average.
"""

import argparse
import collections

import numpy as np
from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

Object = collections.namedtuple('Object', ['label', 'score', 'bbox'])


def tiles_location_gen(img_size, tile_size, overlap):
  """Generates location of tiles after splitting the given image according the tile_size and overlap.

  Args:
    img_size (int, int): size of original image as width x height.
    tile_size (int, int): size of the returned tiles as width x height.
    overlap (int): The number of pixels to overlap the tiles.

  Yields:
    A list of points representing the coordinates of the tile in xmin, ymin,
    xmax, ymax.
  """

  tile_width, tile_height = tile_size
  img_width, img_height = img_size
  h_stride = tile_height - overlap
  w_stride = tile_width - overlap
  for h in range(0, img_height, h_stride):
    for w in range(0, img_width, w_stride):
      xmin = w
      ymin = h
      xmax = min(img_width, w + tile_width)
      ymax = min(img_height, h + tile_height)
      yield [xmin, ymin, xmax, ymax]


def non_max_suppression(objects, threshold):
  """Returns a list of indexes of objects passing the NMS.

  Args:
    objects: result candidates.
    threshold: the threshold of overlapping IoU to merge the boxes.

  Returns:
    A list of indexes containings the objects that pass the NMS.
  """
  if len(objects) == 1:
    return [0]

  boxes = np.array([o.bbox for o in objects])
  xmins = boxes[:, 0]
  ymins = boxes[:, 1]
  xmaxs = boxes[:, 2]
  ymaxs = boxes[:, 3]

  areas = (xmaxs - xmins) * (ymaxs - ymins)
  scores = [o.score for o in objects]
  idxs = np.argsort(scores)

  selected_idxs = []
  while idxs.size != 0:

    selected_idx = idxs[-1]
    selected_idxs.append(selected_idx)

    overlapped_xmins = np.maximum(xmins[selected_idx], xmins[idxs[:-1]])
    overlapped_ymins = np.maximum(ymins[selected_idx], ymins[idxs[:-1]])
    overlapped_xmaxs = np.minimum(xmaxs[selected_idx], xmaxs[idxs[:-1]])
    overlapped_ymaxs = np.minimum(ymaxs[selected_idx], ymaxs[idxs[:-1]])

    w = np.maximum(0, overlapped_xmaxs - overlapped_xmins)
    h = np.maximum(0, overlapped_ymaxs - overlapped_ymins)

    intersections = w * h
    unions = areas[idxs[:-1]] + areas[selected_idx] - intersections
    ious = intersections / unions

    idxs = np.delete(
        idxs, np.concatenate(([len(idxs) - 1], np.where(ious > threshold)[0])))

  return selected_idxs


def draw_object(draw, obj):
  """Draws detection candidate on the image.

  Args:
    draw: the PIL.ImageDraw object that draw on the image.
    obj: The detection candidate.
  """
  draw.rectangle(obj.bbox, outline='red')
  draw.text((obj.bbox[0], obj.bbox[3]), obj.label, fill='#0000')
  draw.text((obj.bbox[0], obj.bbox[3] + 10), str(obj.score), fill='#0000')


def reposition_bounding_box(bbox, tile_location):
  """Relocates bbox to the relative location to the original image.

  Args:
    bbox (int, int, int, int): bounding box relative to tile_location as xmin,
      ymin, xmax, ymax.
    tile_location (int, int, int, int): tile_location in the original image as
      xmin, ymin, xmax, ymax.

  Returns:
    A list of points representing the location of the bounding box relative to
    the original image as xmin, ymin, xmax, ymax.
  """
  bbox[0] = bbox[0] + tile_location[0]
  bbox[1] = bbox[1] + tile_location[1]
  bbox[2] = bbox[2] + tile_location[0]
  bbox[3] = bbox[3] + tile_location[1]
  return bbox


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model',
      required=True,
      help='Detection SSD model path (must have post-processing operator).')
  parser.add_argument('--label', help='Labels file path.')
  parser.add_argument(
      '--score_threshold',
      help='Threshold for returning the candidates.',
      type=float,
      default=0.1)
  parser.add_argument(
      '--tile_sizes',
      help=('Sizes of the tiles to split, could be more than one layer as a '
            'list a with comma delimiter in widthxheight. Example: '
            '"300x300,250x250,.."'),
      required=True)
  parser.add_argument(
      '--tile_overlap',
      help=('Number of pixels to overlap the tiles. tile_overlap should be >= '
            'than half of the min desired object size, otherwise small objects '
            'could be missed on the tile boundary.'),
      type=int,
      default=15)
  parser.add_argument(
      '--iou_threshold',
      help=('threshold to merge bounding box duing nms'),
      type=float,
      default=.1)
  parser.add_argument('--input', help='Input image path.', required=True)
  parser.add_argument('--output', help='Output image path.')
  args = parser.parse_args()

  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()
  labels = read_label_file(args.label) if args.label else {}

  # Open image.
  img = Image.open(args.input).convert('RGB')
  draw = ImageDraw.Draw(img)

  objects_by_label = dict()
  img_size = img.size
  tile_sizes = [
      map(int, tile_size.split('x')) for tile_size in args.tile_sizes.split(',')
  ]
  for tile_size in tile_sizes:
    for tile_location in tiles_location_gen(img_size, tile_size,
                                            args.tile_overlap):
      tile = img.crop(tile_location)
      _, scale = common.set_resized_input(
          interpreter, tile.size,
          lambda size, img=tile: img.resize(size, Image.NEAREST))
      interpreter.invoke()
      objs = detect.get_objects(interpreter, args.score_threshold, scale)

      for obj in objs:
        bbox = [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax]
        bbox = reposition_bounding_box(bbox, tile_location)

        label = labels.get(obj.id, '')
        objects_by_label.setdefault(label,
                                    []).append(Object(label, obj.score, bbox))

  for label, objects in objects_by_label.items():
    idxs = non_max_suppression(objects, args.iou_threshold)
    for idx in idxs:
      draw_object(draw, objects[idx])

  img.show()
  if args.output:
    img.save(args.output)
    print('Done. Results saved at', args.output)


if __name__ == '__main__':
  main()
