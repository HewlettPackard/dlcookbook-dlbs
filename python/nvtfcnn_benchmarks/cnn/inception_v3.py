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
    'batch_size' : 128,
    'data_dir' : None,
    'log_dir' : None,
    'precision' : 'fp16',
    'momentum' : 0.9,
    'learning_rate_init' : 1.0,
    'learning_rate_power' : 2.0,
    'weight_decay' : 1e-4,
    'loss_scale' : 2048.0,
    'larc_eta' : 0.003,
    'larc_mode' : 'clip',
    'num_iter' : 90,
    'iter_unit' : 'epoch',
    'checkpoint_secs' : None,
    'display_every' : 10,
}

args, _ = nvutils.parse_cmdline(default_args)

def inception_v3(inputs, training=False):
    """Google's Inception v3 model
    https://arxiv.org/abs/1512.00567
    """
    def inception_v3_a(builder, x, n):
        cols = [[('conv2d',  64, 1, 1, 'SAME')],
                [('conv2d',  48, 1, 1, 'SAME'), ('conv2d',  64, 5, 1, 'SAME')],
                [('conv2d',  64, 1, 1, 'SAME'), ('conv2d',  96, 3, 1, 'SAME'), ('conv2d',  96, 3, 1, 'SAME')],
                [('apool2d',     3, 1, 'SAME'), ('conv2d',   n, 1, 1, 'SAME')]]
        return builder.inception_module(x, 'incept_v3_a', cols)
    def inception_v3_b(builder, x):
        cols = [[('conv2d',  64, 1, 1, 'SAME'), ('conv2d',  96, 3, 1, 'SAME'), ('conv2d',  96, 3, 2, 'VALID')],
                [('conv2d', 384, 3, 2, 'VALID')],
                [('mpool2d',     3, 2, 'VALID')]]
        return builder.inception_module(x, 'incept_v3_b', cols)
    def inception_v3_c(builder, x, n):
        cols = [[('conv2d', 192, 1, 1, 'SAME')],
                [('conv2d',   n, 1, 1, 'SAME'), ('conv2d',   n, (1,7), 1, 'SAME'), ('conv2d', 192, (7,1), 1, 'SAME')],
                [('conv2d',   n, 1, 1, 'SAME'), ('conv2d',   n, (7,1), 1, 'SAME'), ('conv2d',   n, (1,7), 1, 'SAME'), ('conv2d',   n, (7,1), 1, 'SAME'), ('conv2d', 192, (1,7), 1, 'SAME')],
                [('apool2d',     3, 1, 'SAME'), ('conv2d', 192,    1,  1, 'SAME')]]
        return builder.inception_module(x, 'incept_v3_c', cols)
    def inception_v3_d(builder, x):
        cols = [[('conv2d', 192, 1, 1, 'SAME'), ('conv2d', 320,    3,  2, 'VALID')],
                [('conv2d', 192, 1, 1, 'SAME'), ('conv2d', 192, (1,7), 1, 'SAME'), ('conv2d', 192, (7,1), 1, 'SAME'), ('conv2d', 192, 3, 2, 'VALID')],
                [('mpool2d',     3, 2, 'VALID')]]
        return builder.inception_module(x, 'incept_v3_d',cols)
    def inception_v3_e(builder, x, pooltype):
        poolfunc = {'AVG': 'apool2d', 'MAX': 'mpool2d'}[pooltype]
        cols = [[('conv2d', 320, 1, 1, 'SAME')],
                [('conv2d', 384, 1, 1, 'SAME'), ('conv2d', 384, (1,3), 1, 'SAME')],
                [('share',),                    ('conv2d', 384, (3,1), 1, 'SAME')],
                [('conv2d', 448, 1, 1, 'SAME'), ('conv2d', 384,    3,  1, 'SAME'), ('conv2d', 384, (1,3), 1, 'SAME')],
                [('share',),                    ('share',),                        ('conv2d', 384, (3,1), 1, 'SAME')],
                [(poolfunc,      3, 1, 'SAME'), ('conv2d', 192, 1, 1, 'SAME')]]
        return builder.inception_module(x, 'incept_v3_e', cols)

    # TODO: This does not include the extra 'arm' that forks off
    #         from before the 3rd-last module (the arm is designed
    #         to speed up training in the early stages).
    builder = nvutils.LayerBuilder(tf.nn.relu, args['image_format'], training, use_batch_norm=True)
    x = inputs
    x = builder.conv2d(       x,    32, 3, 2, 'VALID')
    x = builder.conv2d(       x,    32, 3, 1, 'VALID')
    x = builder.conv2d(       x,    64, 3, 1, 'SAME')
    x = builder.max_pooling2d(x,        3, 2, 'VALID')
    x = builder.conv2d(       x,    80, 1, 1, 'VALID')
    x = builder.conv2d(       x,   192, 3, 1, 'VALID')
    x = builder.max_pooling2d(x,        3, 2, 'VALID')
    x = inception_v3_a(builder, x, 32)
    x = inception_v3_a(builder, x, 64)
    x = inception_v3_a(builder, x, 64)
    x = inception_v3_b(builder, x)
    x = inception_v3_c(builder, x, 128)
    x = inception_v3_c(builder, x, 160)
    x = inception_v3_c(builder, x, 160)
    x = inception_v3_c(builder, x, 192)
    x = inception_v3_d(builder, x)
    x = inception_v3_e(builder, x, 'AVG')
    x = inception_v3_e(builder, x, 'MAX')
    return builder.spatial_average2d(x)

nvutils.train(inception_v3, args)

if args['log_dir'] is not None and args['data_dir'] is not None:
    nvutils.validate(inception_v3, args)

