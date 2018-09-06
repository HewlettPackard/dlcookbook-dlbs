#!/usr/bin/env python
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function
from builtins import range
import nvutils
import tensorflow as tf
import argparse

nvutils.init()

default_args = {
    'image_width' : 224,
    'image_height' : 224,
    'image_format' : 'channels_first',
    'distort_color' : False,
    'batch_size' : 256,
    'data_dir' : None,
    'log_dir' : None,
    'precision' : 'fp16',
    'momentum' : 0.9,
    'learning_rate_init' : 0.02,
    'learning_rate_power' : 2.0,
    'weight_decay' : 1e-4,
    'loss_scale' : 128.0,
    'larc_eta' : 0.003,
    'larc_mode' : 'clip',
    'num_iter' : 90,
    'iter_unit' : 'epoch',
    'checkpoint_secs' : None,
    'display_every' : 10,
}

formatter = argparse.ArgumentDefaultsHelpFormatter
parser = argparse.ArgumentParser(formatter_class=formatter)
parser.add_argument('--layers', default=50, type=int, required=True,
                    choices=[11, 13, 16, 19],
                    help="""Number of VGG layers.""")

args, flags = nvutils.parse_cmdline(default_args, parser)

def inference_vgg_impl(builder, inputs, layer_counts):
    x = inputs
    for _ in range(layer_counts[0]): x = builder.conv2d(x,  64, 3, 1, 'SAME')
    x = builder.max_pooling2d(x, 2, 2)
    for _ in range(layer_counts[1]): x = builder.conv2d(x, 128, 3, 1, 'SAME')
    x = builder.max_pooling2d(x, 2, 2)
    for _ in range(layer_counts[2]): x = builder.conv2d(x, 256, 3, 1, 'SAME')
    x = builder.max_pooling2d(x, 2, 2)
    for _ in range(layer_counts[3]): x = builder.conv2d(x, 512, 3, 1, 'SAME')
    x = builder.max_pooling2d(x, 2, 2)
    for _ in range(layer_counts[4]): x = builder.conv2d(x, 512, 3, 1, 'SAME')
    x = builder.max_pooling2d(x, 2, 2)
    x = builder.flatten2d(x)
    x = builder.dense(x, 4096)
    x = builder.dense(x, 4096)
    return x

def vgg(inputs, training=False):
    """Visual Geometry Group's family of models
    https://arxiv.org/abs/1409.1556
    """
    builder = nvutils.LayerBuilder(tf.nn.relu, args['image_format'], training)
    if   flags.layers == 11: return inference_vgg_impl(builder, inputs, [1,1,2,2,2]) # A
    elif flags.layers == 13: return inference_vgg_impl(builder, inputs, [2,2,2,2,2]) # B
    elif flags.layers == 16: return inference_vgg_impl(builder, inputs, [2,2,3,3,3]) # D
    elif flags.layers == 19: return inference_vgg_impl(builder, inputs, [2,2,4,4,4]) # E
    else: raise ValueError("Invalid nlayer (%i); must be one of: 11,13,16,19" %
                           flags.layers)

nvutils.train(vgg, args)

if args['log_dir'] is not None and args['data_dir'] is not None:
    nvutils.validate(vgg, args)

