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
    'image_width' : 299,
    'image_height' : 299,
    'image_format' : 'channels_first',
    'distort_color' : False,
    'batch_size' : 256,
    'data_dir' : None,
    'log_dir' : None,
    'precision' : 'fp16',
    'momentum' : 0.9,
    'learning_rate_init' : 0.045,
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

def inception_v4(inputs, training):
    """Google's Inception v4 model
    https://arxiv.org/abs/1602.07261
    """

    # Stem functions
    def inception_v4_sa(builder, x):
        cols = [[('mpool2d', 3, 2, 'VALID')],
                [('conv2d',  96, 3, 2, 'VALID')]]
        return builder.inception_module(x, 'incept_v4_sa', cols)
    def inception_v4_sb(builder, x):
        cols = [[('conv2d',  64, 1, 1, 'SAME'), ('conv2d',  96, 3, 1, 'VALID')],
                [('conv2d',  64, 1, 1, 'SAME'), ('conv2d',  64, (7,1), 1, 'SAME'), ('conv2d',  64, (1,7), 1, 'SAME'), ('conv2d',  96, 3, 1, 'VALID')]]
        return builder.inception_module(x, 'incept_v4_sb', cols)
    def inception_v4_sc(builder, x):
        cols = [[('conv2d', 192, 3, 2, 'VALID')],
                [('mpool2d', 3, 2, 'VALID')]]
        return builder.inception_module(x, 'incept_v4_sc', cols)

    # Reduction functions
    def inception_v4_ra(builder, x, k, l, m, n):
        cols = [[('mpool2d', 3, 2, 'VALID')],
                [('conv2d',   n, 3, 2, 'VALID')],
                [('conv2d',   k, 1, 1, 'SAME'), ('conv2d',   l, 3, 1, 'SAME'), ('conv2d',   m, 3, 2, 'VALID')]]
        return builder.inception_module(x, 'incept_v4_ra', cols)
    def inception_v4_rb(builder, x):
        cols = [[('mpool2d', 3, 2, 'VALID')],
                [('conv2d', 192, 1, 1, 'SAME'), ('conv2d', 192, 3, 2, 'VALID')],
                [('conv2d', 256, 1, 1, 'SAME'), ('conv2d', 256, (1,7), 1, 'SAME'), ('conv2d', 320, (7,1), 1, 'SAME'), ('conv2d', 320, 3, 2, 'VALID')]]
        return builder.inception_module(x, 'incept_v4_rb', cols)
    def inception_resnet_v2_rb(builder, x):
        cols = [[('mpool2d', 3, 2, 'VALID')],
                # Note: These match Facebook's Torch implem
                [('conv2d', 256, 1, 1, 'SAME'), ('conv2d', 384, 3, 2, 'VALID')],
                [('conv2d', 256, 1, 1, 'SAME'), ('conv2d', 256, 3, 2, 'VALID')],
                [('conv2d', 256, 1, 1, 'SAME'), ('conv2d', 256, 3, 1, 'SAME'), ('conv2d', 256, 3, 2, 'VALID')]]
        return builder.inception_module(x, 'incept_resnet_v2_rb', cols)

    def inception_v4_a(builder, x):
        cols = [[('apool2d', 3, 1, 'SAME'), ('conv2d',  96, 1, 1, 'SAME')],
                [('conv2d',  96, 1, 1, 'SAME')],
                [('conv2d',  64, 1, 1, 'SAME'), ('conv2d',  96, 3, 1, 'SAME')],
                [('conv2d',  64, 1, 1, 'SAME'), ('conv2d',  96, 3, 1, 'SAME'), ('conv2d',  96, 3, 1, 'SAME')]]
        return builder.inception_module(x, 'incept_v4_a', cols)
    def inception_v4_b(builder, x):
        cols = [[('apool2d', 3, 1, 'SAME'), ('conv2d', 128, 1, 1, 'SAME')],
                [('conv2d', 384, 1, 1, 'SAME')],
                [('conv2d', 192, 1, 1, 'SAME'), ('conv2d', 224, (1,7), 1, 'SAME'), ('conv2d', 256, (7,1), 1, 'SAME')],
                [('conv2d', 192, 1, 1, 'SAME'), ('conv2d', 192, (1,7), 1, 'SAME'), ('conv2d', 224, (7,1), 1, 'SAME'), ('conv2d', 224, (1,7), 1, 'SAME'), ('conv2d', 256, (7,1), 1, 'SAME')]]
        return builder.inception_module(x, 'incept_v4_b', cols)
    def inception_v4_c(builder, x):
        cols = [[('apool2d', 3, 1, 'SAME'), ('conv2d', 256, 1, 1, 'SAME')],
                [('conv2d', 256, 1, 1, 'SAME')],
                [('conv2d', 384, 1, 1, 'SAME'), ('conv2d', 256, (1,3), 1, 'SAME')],
                [('share',),           ('conv2d', 256, (3,1), 1, 'SAME')],
                [('conv2d', 384, 1, 1, 'SAME'), ('conv2d', 448, (1,3), 1, 'SAME'), ('conv2d', 512, (3,1), 1, 'SAME'), ('conv2d', 256, (3,1), 1, 'SAME')],
                [('share',),           ('share',),           ('share',),           ('conv2d', 256, (1,3), 1, 'SAME')]]
        return builder.inception_module(x, 'incept_v4_c', cols)

    builder = nvutils.LayerBuilder(tf.nn.relu, args['image_format'], training, use_batch_norm=True)
    x = inputs
    x = builder.conv2d(x, 32, 3, 2, 'VALID')
    x = builder.conv2d(x, 32, 3, 1, 'VALID')
    x = builder.conv2d(x, 64, 3, 1, 'SAME')
    x = inception_v4_sa(builder, x)
    x = inception_v4_sb(builder, x)
    x = inception_v4_sc(builder, x)
    for _ in range(4):
        x = inception_v4_a(builder, x)
    x = inception_v4_ra(builder, x, 192, 224, 256, 384)
    for _ in range(7):
        x = inception_v4_b(builder, x)
    x = inception_v4_rb(builder, x)
    for _ in range(3):
        x = inception_v4_c(builder, x)
    x = builder.spatial_average2d(x)
    x = builder.dropout(x, 0.8)
    return x

nvutils.train(inception_v4, args)

if args['log_dir'] is not None and args['data_dir'] is not None:
    nvutils.validate(inception_v4, args)
