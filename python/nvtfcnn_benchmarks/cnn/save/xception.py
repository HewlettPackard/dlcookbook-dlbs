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
    'batch_size' : 32,
    'data_dir' : None,
    'log_dir' : None,
    'precision' : 'fp32',
    'momentum' : 0.9,
    'learning_rate_init' : 0.045,
    'learning_rate_power' : 2.0,
    'weight_decay' : 4e-5,
    'loss_scale' : 128.0,
    'larc_eta' : 0.003,
    'larc_mode' : 'clip',
    'num_iter' : 90,
    'checkpoint_secs' : None,
    'display_every' : 10,
    'iter_unit' : 'epoch'
}

args, _ = nvutils.parse_cmdline(default_args)

def xception(inputs, training=False):
    builder = nvutils.LayerBuilder(tf.nn.relu, args['image_format'], training, use_batch_norm=True)
    builder.batch_norm_config = {'decay': 0.99, 'epsilon': 1e-5, 'scale': True}
    def make_xception_entry(nout, activate_first=True):
        def xception_entry(inputs):
            x = inputs
            if activate_first:
                x = builder.activate(x)
            x = builder.separable_conv2d(       x, nout, 3, 1, 'SAME')
            x = builder.separable_conv2d_linear(x, nout, 3, 1, 'SAME')
            x = builder.max_pooling2d(          x,       3, 2, 'SAME')
            return x
        return xception_entry
    def xception_middle(inputs):
        x = inputs
        x = builder.activate(x)
        x = builder.separable_conv2d(       x, 728, 3, 1, 'SAME')
        x = builder.separable_conv2d(       x, 728, 3, 1, 'SAME')
        x = builder.separable_conv2d_linear(x, 728, 3, 1, 'SAME')
        return x
    def xception_exit(inputs):
        x = inputs
        x = builder.activate(x)
        x = builder.separable_conv2d(       x,  728, 3, 1, 'SAME')
        x = builder.separable_conv2d_linear(x, 1024, 3, 1, 'SAME')
        x = builder.max_pooling2d(          x,       3, 2, 'SAME')
        return x
    x = inputs
    x = builder.conv2d(x, 32, 3, 2, 'VALID')
    x = builder.conv2d(x, 64, 3, 1, 'VALID')
    x = builder.residual2d(x, make_xception_entry(128, False), 128)
    x = builder.residual2d(x, make_xception_entry(256),        256)
    x = builder.residual2d(x, make_xception_entry(728),        728)
    for _ in range(8):
        x = builder.residual2d(x, xception_middle)
    x = builder.residual2d(x, xception_exit, 1024)
    x = builder.separable_conv2d(x, 1536, 3, 1, 'SAME')
    x = builder.separable_conv2d(x, 2048, 3, 1, 'SAME')
    x = builder.spatial_average2d(x)
    # Note: Optional FC layer not included
    x = builder.dropout(x, 0.5)
    return x

nvutils.train(xception, args)

if args['log_dir'] is not None and args['data_dir'] is not None:
    nvutils.validate(xception, args)

