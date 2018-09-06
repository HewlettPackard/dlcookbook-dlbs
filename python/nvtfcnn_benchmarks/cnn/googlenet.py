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
    'image_width' : 224,
    'image_height' : 224,
    'image_format' : 'channels_first',
    'distort_color' : False,
    'batch_size' : 256,
    'data_dir' : None,
    'log_dir' : None,
    'precision' : 'fp16',
    'momentum' : 0.9,
    'learning_rate_init' : 0.04,
    'learning_rate_power' : 2.0,
    'weight_decay' : 1e-4,
    'loss_scale' : 128.0,
    'larc_eta' : 0.003,
    'larc_mode' : 'clip',
    'num_iter' : 90,
    'checkpoint_secs' : None,
    'display_every' : 10,
    'iter_unit' : 'epoch'
}

args, _ = nvutils.parse_cmdline(default_args)

def googlenet(inputs, training=False):
    """GoogLeNet model
    https://arxiv.org/abs/1409.4842
    """
    def inception_v1(builder, x, k, l, m, n, p, q):
        cols = [[('conv2d',  k, 1, 1, 'SAME')],
                [('conv2d',  l, 1, 1, 'SAME'), ('conv2d', m, 3, 1, 'SAME')],
                [('conv2d',  n, 1, 1, 'SAME'), ('conv2d', p, 5, 1, 'SAME')],
                [('mpool2d',    3, 1, 'SAME'), ('conv2d', q, 1, 1, 'SAME')]]
        return builder.inception_module(x, 'incept_v1', cols)
    builder = nvutils.LayerBuilder(tf.nn.relu, args['image_format'], training)
    x = inputs
    x = builder.conv2d(x,    64, 7, 2, 'SAME')
    x = builder.max_pooling2d(x, 3, 2, 'SAME')
    x = builder.conv2d(x,    64, 1, 1, 'SAME')
    x = builder.conv2d(x,   192, 3, 1, 'SAME')
    x = builder.max_pooling2d(x, 3, 2, 'SAME')
    x = inception_v1(builder, x,  64,  96, 128, 16,  32,  32)
    x = inception_v1(builder, x, 128, 128, 192, 32,  96,  64)
    x = builder.max_pooling2d(x, 3, 2, 'SAME')
    x = inception_v1(builder, x, 192,  96, 208, 16,  48,  64)
    x = inception_v1(builder, x, 160, 112, 224, 24,  64,  64)
    x = inception_v1(builder, x, 128, 128, 256, 24,  64,  64)
    x = inception_v1(builder, x, 112, 144, 288, 32,  64,  64)
    x = inception_v1(builder, x, 256, 160, 320, 32, 128, 128)
    x = builder.max_pooling2d(x, 3, 2, 'SAME')
    x = inception_v1(builder, x, 256, 160, 320, 32, 128, 128)
    x = inception_v1(builder, x, 384, 192, 384, 48, 128, 128)
    x = builder.spatial_average2d(x)
    return x

nvutils.train(googlenet, args)

if args['log_dir'] is not None and args['data_dir'] is not None:
    nvutils.validate(googlenet, args)

