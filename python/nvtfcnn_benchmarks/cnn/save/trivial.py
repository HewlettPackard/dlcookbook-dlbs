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
    'learning_rate_init' : 0.001,
    'learning_rate_power' : 2.0,
    'weight_decay' : 1e-4,
    'loss_scale' : None,
    'larc_eta' : None,
    'larc_mode' : None,
    'num_iter' : 500,
    'checkpoint_secs' : None,
    'display_every' : 10,
    'iter_unit' : 'batch'
}

args, _ = nvutils.parse_cmdline(default_args)

def trivial(inputs, training=False):
    """A trivial model for benchmarking input pipeline performance"""
    builder = nvutils.LayerBuilder(tf.nn.relu, args['image_format'], training)
    x = inputs
    x = builder.flatten2d(x)
    x = builder.dense_linear(x, 1)
    return x

nvutils.train(trivial, args)

if args['log_dir'] is not None and args['data_dir'] is not None:
    nvutils.validate(trivial, args)

