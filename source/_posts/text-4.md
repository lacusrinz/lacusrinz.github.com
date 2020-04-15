---
title: 使用TensorFlow 实现物体识别 -- 自定义数据集训练
date: 2020-02-02 12:34:05
tags: [tensorflow, 物体识别,人工智能]
categories: 机器学习应用
---

## 目标
使用多GPU，在SSDLite with MobileNet-V3-Small backbone模型上训练自定义数据集

<!--more-->

## 环境
Ubuntu 18.04.2 下使用docker镜像：
tensorflow/tensorflow   1.15.0-gpu-py3  
nvidia/cuda             10.1

## 数据集
### 原始照片采集与标注
本文尝试对6张幼儿英语学习卡牌进行识别
卡牌如下
![dataset-1](dataset-1.jpg)
将照片以Class_index格式命名并放入`images`文件夹
分别对每张卡牌拍摄不同场景下的照片，并使用imageLab进行标注，标注为PascalVol格式。标注保存在`annotations`下`xmls`文件夹
![dataset-imageLab](dataset-imagelab.gif)
标注时的类别不是很重要，后面制作tfrecord时候用的是文件名来做类别的
>本文采集了如下数据：
Aa卡牌23张
apple卡牌35张
Be卡牌29张
bee卡牌35张
Cc卡牌15张
cap卡牌38张

### 制作tfrecord
制作`label_map.bptxt`文件，标明class

```xml
item {
  id: 1
  name: 'Aa'
}

item {
  id: 2
  name: 'Bb'
}

item {
  id: 3
  name: 'Cc'
}

item {
  id: 4
  name: 'apple'
}

item {
  id: 5
  name: 'bee'
}

item {
  id: 6
  name: 'cap'
}
```

制作`trainval.txt`，写入所有图片文件名，放在`annotations`文件夹中

```xml
apple_25
Bb_10
Bb_11
Bb_22
bee_29
Bb_3
apple_30
cap_1
bee_10
apple_10
cap_37
apple_17
bee_5
apple_26
Aa_15
cap_2
cap_0
apple_32
apple_34
Bb_6
cap_27
cap_30
bee_4
...等全部文件名
```

在`object_detection/dataset_tools/`下创建`create_abc_tf_record.py`

```python
import hashlib
import io
import logging
import os
import random
import re

import contextlib2
from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/research/abc/data/raw_data', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '/research/abc/data', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', '/research/abc/data/abc_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('faces_only', True, 'If True, generates bounding boxes '
                     'for pet faces.  Otherwise generates bounding boxes (as '
                     'well as segmentations for full pet bodies).  Note that '
                     'in the latter case, the resulting files are much larger.')
flags.DEFINE_string('mask_type', 'png', 'How to represent instance '
                    'segmentation masks. Options are "png" or "numerical".')
flags.DEFINE_integer('num_shards', 1, 'Number of TFRecord shards')

FLAGS = flags.FLAGS

def get_class_name_from_filename(file_name):
  """Gets the class name from a file.

  Args:
    file_name: The file name to get the class name from.
               ie. "american_pit_bull_terrier_105.jpg"

  Returns:
    A string of the class name.
  """
  match = re.match(r'([A-Za-z_]+)(_[0-9]+\.jpg)', file_name, re.I)
  return match.groups()[0]

def dict_to_tf_example(data,
                      #  mask_path,
                       label_map_dict,
                       image_subdirectory,
                       ignore_difficult_instances=False,
                       faces_only=True,
                       mask_type='png'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    mask_path: String path to PNG encoded mask.
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    faces_only: If True, generates bounding boxes for pet faces.  Otherwise
      generates bounding boxes (as well as segmentations for full pet bodies).
    mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
      smaller file sizes.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(image_subdirectory, data['filename'])
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  masks = []
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue
      difficult_obj.append(int(difficult))

      if faces_only:
        xmin = float(obj['bndbox']['xmin'])
        xmax = float(obj['bndbox']['xmax'])
        ymin = float(obj['bndbox']['ymin'])
        ymax = float(obj['bndbox']['ymax'])

      xmins.append(xmin / width)
      ymins.append(ymin / height)
      xmaxs.append(xmax / width)
      ymaxs.append(ymax / height)
      class_name = get_class_name_from_filename(data['filename'])
      classes_text.append(class_name.encode('utf8'))
      classes.append(label_map_dict[class_name])
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))

  feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }


  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example

def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     examples,
                     faces_only=True,
                     mask_type='png'):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    num_shards: Number of shards for output file.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
    faces_only: If True, generates bounding boxes for pet faces.  Otherwise
      generates bounding boxes (as well as segmentations for full pet bodies).
    mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
      smaller file sizes.
  """
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filename, num_shards)
    for idx, example in enumerate(examples):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples))
      xml_path = os.path.join(annotations_dir, 'xmls', example + '.xml')
      # mask_path = os.path.join(annotations_dir, 'trimaps', example + '.png')

      if not os.path.exists(xml_path):
        logging.warning('Could not find %s, ignoring example.', xml_path)
        continue
      with tf.gfile.GFile(xml_path, 'r') as fid:
        xml_str = fid.read()
      xml = etree.fromstring(xml_str)
      data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

      try:
        tf_example = dict_to_tf_example(
            data,
            # mask_path,
            label_map_dict,
            image_dir,
            faces_only=faces_only,
            mask_type=mask_type)
        if tf_example:
          shard_idx = idx % num_shards
          output_tfrecords[shard_idx].write(tf_example.SerializeToString())
      except ValueError:
        logging.warning('Invalid example: %s, ignoring.', xml_path)

def main(_):
  data_dir = FLAGS.data_dir
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  logging.info('Reading from abc dataset.')
  image_dir = os.path.join(data_dir, 'images')
  annotations_dir = os.path.join(data_dir, 'annotations')
  examples_path = os.path.join(annotations_dir, 'trainval.txt')
  examples_list = dataset_util.read_examples_list(examples_path)
  print(examples_list)

  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.
  random.seed(42)
  random.shuffle(examples_list)
  num_examples = len(examples_list)
  num_train = int(0.8 * num_examples)
  train_examples = examples_list[:num_train]
  val_examples = examples_list[num_train:]
  logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))

  train_output_path = os.path.join(FLAGS.output_dir, 'abc_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'abc_val.record')

  create_tf_record(
      train_output_path,
      FLAGS.num_shards,
      label_map_dict,
      annotations_dir,
      image_dir,
      train_examples,
      faces_only=FLAGS.faces_only,
      mask_type=FLAGS.mask_type)
  create_tf_record(
      val_output_path,
      FLAGS.num_shards,
      label_map_dict,
      annotations_dir,
      image_dir,
      val_examples,
      faces_only=FLAGS.faces_only,
      mask_type=FLAGS.mask_type)


if __name__ == '__main__':
  tf.app.run()

```
代码与示例代码`create_pet_tf_record.py`基本一致

运行后获得`abc_train.record-00000-of-00001`和`abc_val.record-00000-of-00001`两个文件


## 训练
### 配置
在[zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)中下载SSDLite with MobileNet-V3-Small backbone模型预训练文件和管道文件，操作方式和文1一样
修改管道文件`ssdlite_mobilenet_v3_small_320x320.config`
```xml
# SSDLite with Mobilenet v3 small feature extractor.
# Trained on COCO14, initialized from scratch.
# TPU-compatible.
# Users should configure the fine_tune_checkpoint field in the train config as
# well as the label_map_path and input_path fields in the train_input_reader and
# eval_input_reader. Search for "PATH_TO_BE_CONFIGURED" to find the fields that
# should be configured.

model {
  ssd {
    inplace_batchnorm_update: true
    freeze_batchnorm: false
    num_classes: 6
    box_coder {
      faster_rcnn_box_coder {
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }
    }
    matcher {
      argmax_matcher {
        matched_threshold: 0.5
        unmatched_threshold: 0.5
        ignore_thresholds: false
        negatives_lower_than_unmatched: true
        force_match_for_each_row: true
        use_matmul_gather: true
      }
    }
    similarity_calculator {
      iou_similarity {
      }
    }
    encode_background_as_zeros: true
    anchor_generator {
      ssd_anchor_generator {
        num_layers: 6
        min_scale: 0.2
        max_scale: 0.95
        aspect_ratios: 1.0
        aspect_ratios: 2.0
        aspect_ratios: 0.5
        aspect_ratios: 3.0
        aspect_ratios: 0.3333
      }
    }
    image_resizer {
      fixed_shape_resizer {
        height: 320
        width: 320
      }
    }
    box_predictor {
      convolutional_box_predictor {
        min_depth: 0
        max_depth: 0
        num_layers_before_predictor: 0
        use_dropout: false
        dropout_keep_probability: 0.8
        kernel_size: 3
        use_depthwise: true
        box_code_size: 4
        apply_sigmoid_to_scores: false
        class_prediction_bias_init: -4.6
        conv_hyperparams {
          activation: RELU_6,
          regularizer {
            l2_regularizer {
              weight: 0.00004
            }
          }
          initializer {
            random_normal_initializer {
              stddev: 0.03
              mean: 0.0
            }
          }
          batch_norm {
            train: true,
            scale: true,
            center: true,
            decay: 0.97,
            epsilon: 0.001,
          }
        }
      }
    }
    feature_extractor {
      type: 'ssd_mobilenet_v3_small'
      min_depth: 16
      depth_multiplier: 1.0
      use_depthwise: true
      conv_hyperparams {
        activation: RELU_6,
        regularizer {
          l2_regularizer {
            weight: 0.00004
          }
        }
        initializer {
          truncated_normal_initializer {
            stddev: 0.03
            mean: 0.0
          }
        }
        batch_norm {
          train: true,
          scale: true,
          center: true,
          decay: 0.97,
          epsilon: 0.001,
        }
      }
      override_base_feature_extractor_hyperparams: true
    }
    loss {
      classification_loss {
        weighted_sigmoid_focal {
          alpha: 0.75,
          gamma: 2.0
        }
      }
      localization_loss {
        weighted_smooth_l1 {
          delta: 1.0
        }
      }
      classification_weight: 1.0
      localization_weight: 1.0
    }
    normalize_loss_by_num_matches: true
    normalize_loc_loss_by_codesize: true
    post_processing {
      batch_non_max_suppression {
        score_threshold: 1e-8
        iou_threshold: 0.6
        max_detections_per_class: 100
        max_total_detections: 100
        use_static_shapes: true
      }
      score_converter: SIGMOID
    }
  }
}

train_config: {
  batch_size: 8
  sync_replicas: true
  startup_delay_steps: 0
  replicas_to_aggregate: 32
  fine_tune_checkpoint: "/research/abc/data/model.ckpt"
  from_detection_checkpoint: true
  load_all_detection_checkpoint_vars: true
  num_steps: 10000
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    ssd_random_crop {
    }
  }
  optimizer {
    momentum_optimizer: {
      learning_rate: {
        cosine_decay_learning_rate {
          learning_rate_base: 0.004
          total_steps: 10000
          warmup_learning_rate: 0.0013333
          warmup_steps: 100
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
  max_number_of_boxes: 100
  unpad_groundtruth_tensors: false
}

train_input_reader: {
  tf_record_input_reader {
    input_path: "/research/abc/data/abc_train*"
  }
  label_map_path: "/research/abc/data/abc_label_map.pbtxt"
}

eval_config: {
  metrics_set: "coco_detection_metrics"
  num_examples: 100
}

eval_input_reader: {
  tf_record_input_reader {
    input_path: "/research/abc/data/abc_val.record*"
  }
  label_map_path: "/research/abc/data/abc_label_map.pbtxt"
  shuffle: false
  num_readers: 1
}

```
>由于本模型不是quantized的，一些op不被支持，所有管道文件最后没有quantized的参数

### tensorboard
![tensorboard_scalars](tensorboard-scalars.png)
9922step后已经达到比较好的效果，随即终止了训练，耗时<1小时
下图为val集上的检测效果
![tensorboard-image1](tensorboard-image1.png)
![tensorboard-image2](tensorboard-image2.png)
![tensorboard-image3](tensorboard-image3.png)

## 转tflite
graph文件转bp与文1一致
由于本模型不是quantized的，所有bp转tflite时需要增加一个参数
```code
allow_custom_ops: Boolean indicating whether to allow custom operations. When false any unknown operation is an error. When true, custom ops are created for any op that is unknown. The developer will need to provide these to the TensorFlow Lite runtime with a custom resolver. (default False)
```
可以使用代码实现
```python
import tensorflow as tf

graph_def_file = "./abc_tflite/tflite_graph.pb"
input_arrays = ["normalized_input_image_tensor"]
output_arrays = ['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']
converter = tf.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays, input_shapes={"normalized_input_image_tensor":[1,320,320,3]})
converter.allow_custom_ops = True
tflite_model = converter.convert()
open("./abc_tflite/detect.tflite", "wb").write(tflite_model)

```
生成tflite文件5.2M

## 移动端实现
与文1一致
效果如下
![tflite](tflite.gif)