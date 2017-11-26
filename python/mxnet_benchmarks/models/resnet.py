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
""" This implementation
    https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/symbols/resnet.py
"""
from __future__ import absolute_import
import mxnet as mx
import numpy as np
from mxnet_benchmarks.models.model import Model

class ResNet(Model):

    implements = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101',
        'resnet152', 'resnet200', 'resnet269'
    ]

    specs = {
        'resnet18':  { 'name': 'ResNet18',  'units': [2, 2, 2, 2],   'num_layers': 18  },  # pylint: disable=C0326
        'resnet34':  { 'name': 'ResNet34',  'units': [3, 4, 6, 3],   'num_layers': 34  },  # pylint: disable=C0326
        'resnet50':  { 'name': 'ResNet50',  'units': [3, 4, 6, 3],   'num_layers': 50  },  # pylint: disable=C0326
        'resnet101': { 'name': 'ResNet101', 'units': [3, 4, 23, 3],  'num_layers': 101 },  # pylint: disable=C0326
        'resnet152': { 'name': 'ResNet152', 'units': [3, 8, 36, 3],  'num_layers': 152 },  # pylint: disable=C0326
        'resnet200': { 'name': 'ResNet200', 'units': [3, 24, 36, 3], 'num_layers': 200 },  # pylint: disable=C0326
        'resnet269': { 'name': 'ResNet269', 'units': [3, 30, 48, 8], 'num_layers': 269 }   # pylint: disable=C0326
    }

    @property
    def output(self):
        return self.__output

    @staticmethod
    def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True,
                      bn_mom=0.9, workspace=256, memonger=False):
        """Return ResNet Unit symbol for building ResNet
        Parameters
        ----------
        data : str
            Input data
        num_filter : int
            Number of output channels
        bnf : int
            Bottle neck channels factor with regard to num_filter
        stride : tuple
            Stride used in convolution
        dim_match : Boolean
            True means channel number between input and output is the same, otherwise means differ
        name : str
            Base name of the operators
        workspace : int
            Workspace used in convolution operator
        """
        if bottle_neck:
            # Branch 1
            if dim_match:
                shortcut = data
            else:
                shortcut = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1),
                                              stride=stride, no_bias=True,
                                              workspace=workspace, name=name+'_sc_conv')
                shortcut = mx.sym.BatchNorm(data=shortcut, fix_gamma=False, eps=2e-5,
                                            momentum=bn_mom, name=name + '_sc_bn')
            if memonger:
                shortcut._set_attr(mirror_stage='True')
            # Branch 2
            #   Block 2A
            conv1 = mx.sym.Convolution(data=data, num_filter=int(num_filter*0.25), kernel=(1, 1),
                                       stride=(1, 1), pad=(0, 0),
                                       no_bias=True, workspace=workspace, name=name + '_conv1')
            bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom,
                                   name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
            #   Block 2B
            conv2 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(3, 3),
                                       stride=stride, pad=(1, 1),
                                       no_bias=True, workspace=workspace, name=name + '_conv2')
            bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom,
                                   name=name + '_bn2')
            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
            #   Block 3B
            conv3 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(1, 1),
                                       stride=(1, 1), pad=(0, 0), no_bias=True,
                                       workspace=workspace, name=name + '_conv3')
            bn3 = mx.sym.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom,
                                   name=name + '_bn3')
            # Element-wise summation and ReLU
            output = shortcut + bn3
            output = mx.sym.Activation(data=output, act_type='relu', name=name + '_relu')
            return output
        else:
            # Branch 1
            if dim_match:
                shortcut = data
            else:
                shortcut = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1, 1),
                                              stride=stride, no_bias=True,
                                              workspace=workspace, name=name+'_sc_conv')
                shortcut = mx.sym.BatchNorm(data=shortcut, fix_gamma=False, momentum=bn_mom,
                                            eps=2e-5, name=name + '_sc_bn')
            if memonger:
                shortcut._set_attr(mirror_stage='True')
            # Branch 2
            #   Block 2A
            conv1 = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3, 3),
                                       stride=stride, pad=(1, 1),
                                       no_bias=True, workspace=workspace, name=name + '_conv1')
            bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5,
                                   name=name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
            #   Block 2B
            conv2 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3),
                                       stride=(1, 1), pad=(1, 1),
                                       no_bias=True, workspace=workspace, name=name + '_conv2')
            bn2 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5,
                                   name=name + '_bn2')

            # Element-wise summation and ReLU
            output = shortcut + bn2
            output = mx.sym.Activation(data=output, act_type='relu', name=name + '_relu')
            return output

    def resnet(self, units, num_stages, filter_list, bottle_neck=True, bn_mom=0.9,
               workspace=256, dtype='float32', memonger=False):
        """Return ResNet symbol of
        Parameters
        ----------
        units : list
            Number of units in each stage
        num_stages : int
            Number of stage
        filter_list : list
            Channel size of each stage
        num_classes : int
            Ouput size of symbol
        dataset : str
            Dataset type, only cifar10 and imagenet supports
        workspace : int
            Workspace used in convolution operator
        dtype : str
            Precision (float32 or float16)
        """
        num_unit = len(units)
        assert num_unit == num_stages
        v = mx.sym.Variable(name='data')
        if dtype == 'float32':
            v = mx.sym.identity(data=v, name='id')
        else:
            if dtype == 'float16':
                v = mx.sym.Cast(data=v, dtype=np.float16)

        v = mx.sym.Convolution(data=v, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2),
                               pad=(3, 3), no_bias=True, name="conv0", workspace=workspace)
        v = mx.sym.BatchNorm(data=v, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        v = mx.sym.Activation(data=v, act_type='relu', name='relu0')
        v = mx.sym.Pooling(data=v, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')

        for i in range(num_stages):
            v = ResNet.residual_unit(v, filter_list[i+1], (1 if i == 0 else 2, 1 if i == 0 else 2),
                                     False, name='stage%d_unit%d' % (i + 1, 1),
                                     bottle_neck=bottle_neck, workspace=workspace,
                                     memonger=memonger)
            for j in range(units[i]-1):
                v = ResNet.residual_unit(v, filter_list[i+1], (1, 1), True,
                                         name='stage%d_unit%d' % (i + 1, j + 2),
                                         bottle_neck=bottle_neck, workspace=workspace,
                                         memonger=memonger)

        # Although kernel is not used here when global_pool=True, we should put one
        v = mx.sym.Pooling(data=v, global_pool=True, kernel=(7, 7), pool_type='avg',
                           name='pool1')
        v = mx.sym.Flatten(data=v)
        v = self.add_head_nodes(v)
        return v

    def __init__(self, params):
        specs = ResNet.specs[params['model']]
        Model.check_parameters(
            params,
            {'name': specs['name'], 'input_shape':(3, 224, 224),
             'num_classes': 1000, 'phase': 'training',
             'dtype': 'float32'}
        )
        Model.__init__(self, params)
        # Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
        # Original author Wei Wu
        if specs['num_layers'] >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False

        self.__output = self.resnet(
            units=specs['units'],
            num_stages=4,
            filter_list=filter_list,
            bottle_neck=bottle_neck,
            workspace=256,
            dtype=params['dtype']
        )



# Original (buggy) implementation not consistent with reference Caffe's descriptors
#class ResNet(Model):
#
#    implements = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200', 'resnet269']
#
#    specs = {
#        'resnet18':  { 'name': 'ResNet18',  'units': [2, 2, 2, 2],   'num_layers': 18  },  # pylint: disable=C0326
#        'resnet34':  { 'name': 'ResNet34',  'units': [3, 4, 6, 3],   'num_layers': 34  },  # pylint: disable=C0326
#        'resnet50':  { 'name': 'ResNet50',  'units': [3, 4, 6, 3],   'num_layers': 50  },  # pylint: disable=C0326
#        'resnet101': { 'name': 'ResNet101', 'units': [3, 4, 23, 3],  'num_layers': 101 },  # pylint: disable=C0326
#        'resnet152': { 'name': 'ResNet152', 'units': [3, 8, 36, 3],  'num_layers': 152 },  # pylint: disable=C0326
#        'resnet200': { 'name': 'ResNet200', 'units': [3, 24, 36, 3], 'num_layers': 200 },  # pylint: disable=C0326
#        'resnet269': { 'name': 'ResNet269', 'units': [3, 30, 48, 8], 'num_layers': 269 }   # pylint: disable=C0326
#    }
#
#    @property
#    def output(self):
#        return self.__output
#
#    def residual_unit(self, data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
#        """Return ResNet Unit symbol for building ResNet
#        Parameters
#        ----------
#        data : str
#            Input data
#        num_filter : int
#            Number of output channels
#        bnf : int
#            Bottle neck channels factor with regard to num_filter
#        stride : tuple
#            Stride used in convolution
#        dim_match : Boolean
#            True means channel number between input and output is the same, otherwise means differ
#        name : str
#            Base name of the operators
#        workspace : int
#            Workspace used in convolution operator
#        """
#        if bottle_neck:
#            # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
#            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
#            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
#            conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
#                                       no_bias=True, workspace=workspace, name=name + '_conv1')
#            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
#            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
#            conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
#                                       no_bias=True, workspace=workspace, name=name + '_conv2')
#            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
#            act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
#            conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
#                                       workspace=workspace, name=name + '_conv3')
#            if dim_match:
#                shortcut = data
#            else:
#                shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
#                                              workspace=workspace, name=name+'_sc')
#            if memonger:
#                shortcut._set_attr(mirror_stage='True')
#            return conv3 + shortcut
#        else:
#            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
#            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
#            conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
#                                       no_bias=True, workspace=workspace, name=name + '_conv1')
#            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
#            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
#            conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
#                                       no_bias=True, workspace=workspace, name=name + '_conv2')
#            if dim_match:
#                shortcut = data
#            else:
#                shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
#                                                workspace=workspace, name=name+'_sc')
#            if memonger:
#                shortcut._set_attr(mirror_stage='True')
#            return conv2 + shortcut
#
#    def resnet(self, units, num_stages, filter_list, num_classes, image_shape, bottle_neck=True, bn_mom=0.9, workspace=256, dtype='float32', memonger=False):
#        """Return ResNet symbol of
#        Parameters
#        ----------
#        units : list
#            Number of units in each stage
#        num_stages : int
#            Number of stage
#        filter_list : list
#            Channel size of each stage
#        num_classes : int
#            Ouput size of symbol
#        dataset : str
#            Dataset type, only cifar10 and imagenet supports
#        workspace : int
#            Workspace used in convolution operator
#        dtype : str
#            Precision (float32 or float16)
#        """
#        num_unit = len(units)
#        assert num_unit == num_stages
#        data = mx.sym.Variable(name='data')
#        if dtype == 'float32':
#            data = mx.sym.identity(data=data, name='id')
#        else:
#            if dtype == 'float16':
#                data = mx.sym.Cast(data=data, dtype=np.float16)
#        data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
#        (nchannel, height, width) = image_shape
#        if height <= 32:            # such as cifar10
#            body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
#                                      no_bias=True, name="conv0", workspace=workspace)
#        else:                       # often expected to be 224 such as imagenet
#            body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
#                                      no_bias=True, name="conv0", workspace=workspace)
#            body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
#            body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
#            body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
#
#        for i in range(num_stages):
#            body = self.residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
#                                      name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
#                                      memonger=memonger)
#            for j in range(units[i]-1):
#                body = self.residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
#                                          bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
#        bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
#        relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
#        # Although kernel is not used here when global_pool=True, we should put one
#        pool1 = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
#        flat = mx.sym.Flatten(data=pool1)
#        fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
#        if dtype == 'float16':
#            fc1 = mx.sym.Cast(data=fc1, dtype=np.float32)
#
#        if self.phase == 'training':
#            labels = mx.sym.Variable(name="softmax_label")
#            v = mx.symbol.SoftmaxOutput(data=fc1, label=labels, name='softmax')
#        else:
#            v = mx.symbol.softmax(data=fc1, name='softmax')
#        return v
#
#    def __init__(self, params):
#        specs = ResNet.specs[params['model']]
#        Model.check_parameters(
#            params,
#            {'name': specs['name'], 'input_shape':(3, 224, 224),
#             'num_classes': 1000, 'phase': 'training'}
#        )
#        Model.__init__(self, params)
#        # Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
#        # Original author Wei Wu
#        (nchannel, height, width) = self.input_shape
#        num_layers = specs['num_layers']
#        conv_workspace=256
#        dtype='float32'
#        if height <= 28:
#            num_stages = 3
#            if (num_layers-2) % 9 == 0 and num_layers >= 164:
#                per_unit = [(num_layers-2)//9]
#                filter_list = [16, 64, 128, 256]
#                bottle_neck = True
#            elif (num_layers-2) % 6 == 0 and num_layers < 164:
#                per_unit = [(num_layers-2)//6]
#                filter_list = [16, 16, 32, 64]
#                bottle_neck = False
#            else:
#                raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
#            units = per_unit * num_stages
#        else:
#            if num_layers >= 50:
#                filter_list = [64, 256, 512, 1024, 2048]
#                bottle_neck = True
#            else:
#                filter_list = [64, 64, 128, 256, 512]
#                bottle_neck = False
#            num_stages = 4
#            units = specs['units']
#
#        self.__output = self.resnet(
#            units=units,
#            num_stages=num_stages,
#            filter_list=filter_list,
#            num_classes=self.num_classes,
#            image_shape=self.input_shape,
#            bottle_neck=bottle_neck,
#            workspace=conv_workspace,
#            dtype=dtype
#        )
