# (c) Copyright [2017] Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Naming and topology according to: http://ethereon.github.io/netscope/#/gist/5c94a074f4e4ac4b81ee28a796e04b5d
    Based on: https://github.com/dmlc/mxnet/blob/master/example/image-classification/symbols/alexnet.py
    Reference AlexNet with grouped convolutions removed.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mxnet as mx
from mxnet_benchmarks.models.model import Model, Layers


class AlexNet(Model):

    implements = 'alexnet'

    @property
    def output(self):
        return self.__output

    def __init__(self, params):
        Model.check_parameters(
            params,
            {'name': 'AlexNet', 'num_classes': 1000, 'phase': 'training', 'dtype': 'float32',
             'input_layout': 'NCHW', 'model_layout': 'NCHW', 'nvidia_layers': False}
        )
        params['input_shape'] = Model.conv_shape(3, (227, 227), params['input_layout'])
        Model.__init__(self, params)

        layers = Layers(params)

        data = self.add_data_node()
        data = Layers.conv_transform_layout(data, params['input_layout'], params['model_layout'])

        conv1 = layers.Convolution(name='conv1', data=data, kernel=(11, 11), stride=(4, 4), num_filter=96)
        relu1 = layers.Activation(name='relu1', data=conv1, act_type='relu')
        norm1 = self.maybe_lrn(relu1, 'norm1')
        pool1 = layers.Pooling(name='pool1', data=norm1, pool_type="max", kernel=(3, 3), stride=(2, 2))

        conv2 = layers.Convolution(name='conv2', data=pool1, kernel=(5, 5), pad=(2, 2), num_filter=256, num_group=1)
        relu2 = layers.Activation(name='relu2', data=conv2, act_type="relu")
        norm2 = self.maybe_lrn(relu2, 'norm2')
        pool2 = layers.Pooling(name='pool2', data=norm2, kernel=(3, 3), stride=(2, 2), pool_type="max")

        conv3 = layers.Convolution(name='conv3', data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=384)
        relu3 = layers.Activation(name='relu3', data=conv3, act_type="relu")

        conv4 = layers.Convolution(name='conv4', data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384, num_group=1)
        relu4 = layers.Activation(name='relu4', data=conv4, act_type="relu")

        conv5 = layers.Convolution(name='conv5', data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256, num_group=1)
        relu5 = layers.Activation(name='relu5', data=conv5, act_type="relu")
        pool5 = layers.Pooling(name='pool5', data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")

        flatten = mx.symbol.Flatten(data=pool5)
        fc6 = mx.symbol.FullyConnected(name='fc6', data=flatten, num_hidden=4096)
        relu6 = layers.Activation(name='relu6', data=fc6, act_type="relu")
        drop6 = layers.Dropout(name='drop6', data=relu6, p=0.5)

        fc7 = mx.symbol.FullyConnected(name='fc7', data=drop6, num_hidden=4096)
        relu7 = layers.Activation(name='relu7', data=fc7, act_type="relu")
        drop7 = layers.Dropout(name='drop7', data=relu7, p=0.5)

        self.__output = self.add_head_nodes(drop7)
