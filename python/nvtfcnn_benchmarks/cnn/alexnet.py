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

nvutils.init()

default_args = {
    'image_width' : 227,
    'image_height' : 227,
    'image_format' : 'channels_first',
    'distort_color' : False,
    'batch_size' : 256,
    'data_dir' : None,
    'log_dir' : None,
    'precision' : 'fp16',
    'momentum' : 0.9,
    'learning_rate_init' : 1.0,
    'learning_rate_power' : 2.0,
    'weight_decay' : 5e-4,
    'loss_scale' : 128.0,
    'larc_eta' : 0.0023,
    'larc_mode' : 'clip',
    'num_iter' : 91,
    'checkpoint_secs' : None,
    'display_every' : 10,
    'iter_unit' : 'epoch'
}

args, _ = nvutils.parse_cmdline(default_args)

def alexnet_owt(inputs, training):
    """Alexnet One Weird Trick model
    https://arxiv.org/abs/1404.5997
    """
    builder = nvutils.LayerBuilder(tf.nn.relu, args['image_format'], training)
    # Note: VALID requires padding the images by 3 in width and height
    x = inputs
    x = builder.conv2d(x, 64, 11, 4, 'VALID')
    x = builder.max_pooling2d(x, 3, 2)
    x = builder.conv2d(x, 192,   5, 1, 'SAME')
    x = builder.max_pooling2d(x, 3, 2)
    x = builder.conv2d(x, 384,   3, 1, 'SAME')
    x = builder.conv2d(x, 256,   3, 1, 'SAME')
    x = builder.conv2d(x, 256,   3, 1, 'SAME')
    x = builder.max_pooling2d(x, 3, 2)
    x = builder.flatten2d(x)
    x = builder.dense(x, 4096)
    x = builder.dropout(x)
    x = builder.dense(x, 4096)
    x = builder.dropout(x)
    return x

nvutils.train(alexnet_owt, args)

if args['log_dir'] is not None and args['data_dir'] is not None:
    nvutils.validate(alexnet_owt, args)

