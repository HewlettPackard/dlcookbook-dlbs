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

def inception_resnet_v2(inputs, training):
    """Google's Inception-Resnet v2 model
    https://arxiv.org/abs/1602.07261
    """
    builder = nvutils.LayerBuilder(tf.nn.relu, args['image_format'], training, use_batch_norm=True)

    # Stem functions
    def inception_v4_sa(x):
        cols = [[('mpool2d', 3, 2, 'VALID')],
                [('conv2d',  96, 3, 2, 'VALID')]]
        return builder.inception_module(x, 'incept_v4_sa', cols)
    def inception_v4_sb(x):
        cols = [[('conv2d',  64, 1, 1, 'SAME'), ('conv2d',  96, 3, 1, 'VALID')],
                [('conv2d',  64, 1, 1, 'SAME'), ('conv2d',  64, (7,1), 1, 'SAME'), ('conv2d',  64, (1,7), 1, 'SAME'), ('conv2d',  96, 3, 1, 'VALID')]]
        return builder.inception_module(x, 'incept_v4_sb', cols)
    def inception_v4_sc(x):
        cols = [[('conv2d', 192, 3, 2, 'VALID')],
                [('mpool2d', 3, 2, 'VALID')]]
        return builder.inception_module(x, 'incept_v4_sc', cols)

    # Reduction functions
    def inception_v4_ra(x, k, l, m, n):
        cols = [[('mpool2d', 3, 2, 'VALID')],
                [('conv2d',   n, 3, 2, 'VALID')],
                [('conv2d',   k, 1, 1, 'SAME'), ('conv2d',   l, 3, 1, 'SAME'), ('conv2d',   m, 3, 2, 'VALID')]]
        return builder.inception_module(x, 'incept_v4_ra', cols)
    def inception_v4_rb(x):
        cols = [[('mpool2d', 3, 2, 'VALID')],
                [('conv2d', 192, 1, 1, 'SAME'), ('conv2d', 192, 3, 2, 'VALID')],
                [('conv2d', 256, 1, 1, 'SAME'), ('conv2d', 256, (1,7), 1, 'SAME'), ('conv2d', 320, (7,1), 1, 'SAME'), ('conv2d', 320, 3, 2, 'VALID')]]
        return builder.inception_module(x, 'incept_v4_rb', cols)
    def inception_resnet_v2_rb(x):
        cols = [[('mpool2d', 3, 2, 'VALID')],
                # Note: These match Facebook's Torch implem
                [('conv2d', 256, 1, 1, 'SAME'), ('conv2d', 384, 3, 2, 'VALID')],
                [('conv2d', 256, 1, 1, 'SAME'), ('conv2d', 256, 3, 2, 'VALID')],
                [('conv2d', 256, 1, 1, 'SAME'), ('conv2d', 256, 3, 1, 'SAME'), ('conv2d', 256, 3, 2, 'VALID')]]
        return builder.inception_module(x, 'incept_resnet_v2_rb', cols)

    def inception_resnet_v2_a(x):
        cols = [[('conv2d',  32, 1, 1, 'SAME')],
                [('conv2d',  32, 1, 1, 'SAME'), ('conv2d',  32, 3, 1, 'SAME')],
                [('conv2d',  32, 1, 1, 'SAME'), ('conv2d',  48, 3, 1, 'SAME'), ('conv2d',  64, 3, 1, 'SAME')]]
        x = builder.inception_module(x, 'incept_resnet_v2_a', cols)
        x = builder.conv2d_linear(x, 384, 1, 1, 'SAME')
        return x
    def inception_resnet_v2_b(x):
        cols = [[('conv2d', 192, 1, 1, 'SAME')],
                [('conv2d', 128, 1, 1, 'SAME'), ('conv2d', 160, (1,7), 1, 'SAME'), ('conv2d', 192, (7,1), 1, 'SAME')]]
        x = builder.inception_module(x, 'incept_resnet_v2_b', cols)
        x = builder.conv2d_linear(x, 1152, 1, 1, 'SAME')
        return x
    def inception_resnet_v2_c(x):
        cols = [[('conv2d', 192, 1, 1, 'SAME')],
                [('conv2d', 192, 1, 1, 'SAME'), ('conv2d', 224, (1,3), 1, 'SAME'), ('conv2d', 256, (3,1), 1, 'SAME')]]
        x = builder.inception_module(x, 'incept_resnet_v2_c', cols)
        x = builder.conv2d_linear(x, 2048, 1, 1, 'SAME')
        return x

    residual_scale = 0.2
    x = inputs
    x = builder.conv2d(x, 32, 3, 2, 'VALID')
    x = builder.conv2d(x, 32, 3, 1, 'VALID')
    x = builder.conv2d(x, 64, 3, 1, 'SAME')
    x = inception_v4_sa(x)
    x = inception_v4_sb(x)
    x = inception_v4_sc(x)
    for _ in range(5):
        x = builder.residual2d(x, inception_resnet_v2_a, scale=residual_scale)
    x = inception_v4_ra(x, 256, 256, 384, 384)
    for _ in range(10):
        x = builder.residual2d(x, inception_resnet_v2_b, scale=residual_scale)
    x = inception_resnet_v2_rb(x)
    for _ in range(5):
        x = builder.residual2d(x, inception_resnet_v2_c, scale=residual_scale)
    x = builder.spatial_average2d(x)
    x = builder.dropout(x, 0.8)
    return x

nvutils.train(inception_resnet_v2, args)

if args['log_dir'] is not None and args['data_dir'] is not None:
    nvutils.validate(inception_resnet_v2, args)

