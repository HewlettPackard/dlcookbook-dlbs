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
""" 
    Architecture described here https://arxiv.org/pdf/1312.6229.pdf
    Based on Google's tf_cnn_benchmark implementation with dropout applied to
    fully connected layers as described in the paper.
"""
from __future__ import absolute_import
import mxnet as mx
from mxnet_benchmarks.models.model import Model

class Overfeat(Model):

    implements = 'overfeat'

    @property
    def output(self):
        return self.__output

    def __init__(self, params):
        Model.check_parameters(
            params,
            {'name': 'Overfeat', 'input_shape': (3, 231, 231), 'num_classes': 1000,
             'phase': 'training',
             'dtype': 'float32'}
        )
        Model.__init__(self, params)
        training = self.phase == 'training'

        data = self.add_data_node()
        # Layer1
        conv1 = mx.symbol.Convolution(name='conv1', data=data, kernel=(11, 11), stride=(4, 4), num_filter=96)
        relu1 = mx.symbol.Activation(name='relu1', data=conv1, act_type='relu')
        pool1 = mx.symbol.Pooling(name='pool1', data=relu1, pool_type="max", kernel=(2, 2), stride=(2, 2))
        # Layer2
        conv2 = mx.symbol.Convolution(name='conv2', data=pool1, kernel=(5, 5), num_filter=256)
        relu2 = mx.symbol.Activation(name='relu2', data=conv2, act_type="relu")
        pool2 = mx.symbol.Pooling(name='pool2', data=relu2, kernel=(2, 2), stride=(2, 2), pool_type="max")
        # Layer3
        conv3 = mx.symbol.Convolution(name='conv3', data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=512)
        relu3 = mx.symbol.Activation(name='relu3', data=conv3, act_type="relu")
        # Layer4
        conv4 = mx.symbol.Convolution(name='conv4', data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=1024)
        relu4 = mx.symbol.Activation(name='relu4', data=conv4, act_type="relu")
        # Layer5
        conv5 = mx.symbol.Convolution(name='conv5', data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=1024)
        relu5 = mx.symbol.Activation(name='relu5', data=conv5, act_type="relu")
        pool5 = mx.symbol.Pooling(name='pool5', data=relu5, kernel=(2, 2), stride=(2, 2), pool_type="max")
        # Layer6
        flatten = mx.symbol.Flatten(data=pool5)
        fc6 = mx.symbol.FullyConnected(name='fc6', data=flatten, num_hidden=3072)
        relu6 = mx.symbol.Activation(name='relu6', data=fc6, act_type="relu")
        drop6 = mx.symbol.Dropout(name='drop6', data=relu6, p=0.5) if training else relu6
        # Layer7
        fc7 = mx.symbol.FullyConnected(name='fc7', data=drop6, num_hidden=4096)
        relu7 = mx.symbol.Activation(name='relu7', data=fc7, act_type="relu")
        drop7 = mx.symbol.Dropout(name='drop7', data=relu7, p=0.5) if training else relu7

        self.__output = self.add_head_nodes(drop7)
