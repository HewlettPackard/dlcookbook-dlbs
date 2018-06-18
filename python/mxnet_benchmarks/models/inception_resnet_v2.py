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
"""Basically, this implementation:
https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/symbols/inception-resnet-v2.py

Visualization (may be slightly different from this implementation):
http://ethereon.github.io/netscope/#/gist/95826c28109d1a5ab95306c37bb30225
"""
from __future__ import absolute_import
import mxnet as mx
from mxnet_benchmarks.models.model import Model

def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0),
                act_type="relu", mirror_attr=None, with_act=True):
    mirror_attr = mirror_attr or {}
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter,
                                 kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv)
    if with_act:
        act = mx.symbol.Activation(
            data=bn, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return bn


def block35(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr=None):
    mirror_attr = mirror_attr or {}
    tower_conv = ConvFactory(net, 32, (1, 1))
    tower_conv1_0 = ConvFactory(net, 32, (1, 1))
    tower_conv1_1 = ConvFactory(tower_conv1_0, 32, (3, 3), pad=(1, 1))
    tower_conv2_0 = ConvFactory(net, 32, (1, 1))
    tower_conv2_1 = ConvFactory(tower_conv2_0, 48, (3, 3), pad=(1, 1))
    tower_conv2_2 = ConvFactory(tower_conv2_1, 64, (3, 3), pad=(1, 1))
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_1, tower_conv2_2])
    tower_out = ConvFactory(
        tower_mixed, input_num_channels, (1, 1), with_act=False)

    net = net + scale * tower_out
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net


def block17(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr=None):
    mirror_attr = mirror_attr or {}
    tower_conv = ConvFactory(net, 192, (1, 1))
    tower_conv1_0 = ConvFactory(net, 129, (1, 1))
    tower_conv1_1 = ConvFactory(tower_conv1_0, 160, (1, 7), pad=(1, 2))
    tower_conv1_2 = ConvFactory(tower_conv1_1, 192, (7, 1), pad=(2, 1))
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
    tower_out = ConvFactory(
        tower_mixed, input_num_channels, (1, 1), with_act=False)
    net = net + scale * tower_out
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net


def block8(net, input_num_channels, scale=1.0, with_act=True, act_type='relu', mirror_attr=None):
    mirror_attr = mirror_attr or {}
    tower_conv = ConvFactory(net, 192, (1, 1))
    tower_conv1_0 = ConvFactory(net, 192, (1, 1))
    tower_conv1_1 = ConvFactory(tower_conv1_0, 224, (1, 3), pad=(0, 1))
    tower_conv1_2 = ConvFactory(tower_conv1_1, 256, (3, 1), pad=(1, 0))
    tower_mixed = mx.symbol.Concat(*[tower_conv, tower_conv1_2])
    tower_out = ConvFactory(
        tower_mixed, input_num_channels, (1, 1), with_act=False)
    net = net + scale * tower_out
    if with_act:
        act = mx.symbol.Activation(
            data=net, act_type=act_type, attr=mirror_attr)
        return act
    else:
        return net


def repeat(inputs, repetitions, layer, *args, **kwargs):
    outputs = inputs
    for _ in range(repetitions):
        outputs = layer(outputs, *args, **kwargs)
    return outputs


class InceptionResNetV2(Model):

    implements = 'inception_resnet_v2'

    @property
    def output(self):
        return self.__output

    def __init__(self, params):
        """ """
        Model.check_parameters(
            params,
            {'name': 'InceptionResNetV2', 'input_shape':(3, 299, 299),
             'num_classes': 1000, 'phase': 'training',
             'dtype': 'float32'}
        )
        Model.__init__(self, params)

        data = self.add_data_node()
        conv1a_3_3 = ConvFactory(data=data, num_filter=32, kernel=(3, 3),
                                 stride=(2, 2))
        conv2a_3_3 = ConvFactory(conv1a_3_3, 32, (3, 3))
        conv2b_3_3 = ConvFactory(conv2a_3_3, 64, (3, 3), pad=(1, 1))
        maxpool3a_3_3 = mx.symbol.Pooling(data=conv2b_3_3, kernel=(3, 3),
                                          stride=(2, 2), pool_type='max')

        conv3b_1_1 = ConvFactory(maxpool3a_3_3, 80, (1, 1))
        conv4a_3_3 = ConvFactory(conv3b_1_1, 192, (3, 3))
        maxpool5a_3_3 = mx.symbol.Pooling(data=conv4a_3_3, kernel=(3, 3),
                                          stride=(2, 2), pool_type='max')

        tower_conv = ConvFactory(maxpool5a_3_3, 96, (1, 1))
        tower_conv1_0 = ConvFactory(maxpool5a_3_3, 48, (1, 1))
        tower_conv1_1 = ConvFactory(tower_conv1_0, 64, (5, 5), pad=(2, 2))

        tower_conv2_0 = ConvFactory(maxpool5a_3_3, 64, (1, 1))
        tower_conv2_1 = ConvFactory(tower_conv2_0, 96, (3, 3), pad=(1, 1))
        tower_conv2_2 = ConvFactory(tower_conv2_1, 96, (3, 3), pad=(1, 1))

        tower_pool3_0 = mx.symbol.Pooling(data=maxpool5a_3_3, kernel=(
            3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg')
        tower_conv3_1 = ConvFactory(tower_pool3_0, 64, (1, 1))
        tower_5b_out = mx.symbol.Concat(
            *[tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_1])
        net = repeat(tower_5b_out, 10, block35, scale=0.17, input_num_channels=320)
        tower_conv = ConvFactory(net, 384, (3, 3), stride=(2, 2))
        tower_conv1_0 = ConvFactory(net, 256, (1, 1))
        tower_conv1_1 = ConvFactory(tower_conv1_0, 256, (3, 3), pad=(1, 1))
        tower_conv1_2 = ConvFactory(tower_conv1_1, 384, (3, 3), stride=(2, 2))
        tower_pool = mx.symbol.Pooling(net, kernel=(
            3, 3), stride=(2, 2), pool_type='max')
        net = mx.symbol.Concat(*[tower_conv, tower_conv1_2, tower_pool])
        net = repeat(net, 20, block17, scale=0.1, input_num_channels=1088)
        tower_conv = ConvFactory(net, 256, (1, 1))
        tower_conv0_1 = ConvFactory(tower_conv, 384, (3, 3), stride=(2, 2))
        tower_conv1 = ConvFactory(net, 256, (1, 1))
        tower_conv1_1 = ConvFactory(tower_conv1, 288, (3, 3), stride=(2, 2))
        tower_conv2 = ConvFactory(net, 256, (1, 1))
        tower_conv2_1 = ConvFactory(tower_conv2, 288, (3, 3), pad=(1, 1))
        tower_conv2_2 = ConvFactory(tower_conv2_1, 320, (3, 3), stride=(2, 2))
        tower_pool = mx.symbol.Pooling(net, kernel=(3, 3), stride=(2, 2), pool_type='max')
        net = mx.symbol.Concat(
            *[tower_conv0_1, tower_conv1_1, tower_conv2_2, tower_pool])

        net = repeat(net, 9, block8, scale=0.2, input_num_channels=2080)
        net = block8(net, with_act=False, input_num_channels=2080)

        net = ConvFactory(net, 1536, (1, 1))
        net = mx.symbol.Pooling(net, kernel=(
            1, 1), global_pool=True, stride=(2, 2), pool_type='avg')
        net = mx.symbol.Flatten(net)
        if self.phase == 'training':
            net = mx.symbol.Dropout(data=net, p=0.2)
        self.__output = self.add_head_nodes(net)
