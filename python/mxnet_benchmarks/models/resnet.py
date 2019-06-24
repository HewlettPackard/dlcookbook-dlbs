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
from __future__ import division
from __future__ import print_function
import mxnet as mx
from mxnet_benchmarks.models.model import Model, Layers


class ResNet(Model):

    implements = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101',
        'resnet152', 'resnet200', 'resnet269'
    ]

    specs = {
        'resnet18':  {'name': 'ResNet18',  'units': [2, 2, 2, 2],   'num_layers': 18},   # pylint: disable=C0326
        'resnet34':  {'name': 'ResNet34',  'units': [3, 4, 6, 3],   'num_layers': 34},   # pylint: disable=C0326
        'resnet50':  {'name': 'ResNet50',  'units': [3, 4, 6, 3],   'num_layers': 50},   # pylint: disable=C0326
        'resnet101': {'name': 'ResNet101', 'units': [3, 4, 23, 3],  'num_layers': 101},  # pylint: disable=C0326
        'resnet152': {'name': 'ResNet152', 'units': [3, 8, 36, 3],  'num_layers': 152},  # pylint: disable=C0326
        'resnet200': {'name': 'ResNet200', 'units': [3, 24, 36, 3], 'num_layers': 200},  # pylint: disable=C0326
        'resnet269': {'name': 'ResNet269', 'units': [3, 30, 48, 8], 'num_layers': 269}   # pylint: disable=C0326
    }

    @property
    def output(self):
        return self.__output

    def residual_unit(self, data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256,
                      memonger=False, cudnn_bn_off=False, fuse_bn_relu=False, fuse_bn_add_relu=False):
        """Return ResNet Unit symbol for building ResNet

        Args:
            data: Input tensor to this residual unit.
            num_filter (int): Number of filters.
            stride (tuple): Stride used in convolution
            dim_match (boolean): True means channel number between input and output is the same, otherwise means differ.
            name (str): Base name of the module
            workspace (int): Workspace used in convolution operator
            cudnn_bn_off (boolean): Do not use CUDNN for batch norm operator
            fuse_bn_relu (boolean): Use fused implementation of batch norm and activation, only available in NGC
                containers.
            fuse_bn_add_relu (boolean): Use fused implementation. Only available in NGC containers.
        Returns:
            Output tensor

        Since new layer factory is used, depending on the runtime (like specific NGC containers), neural net operators
        may have additional parameters.
        """
        act = 'relu' if fuse_bn_relu else None
        if bottle_neck:
            # Branch 1
            if dim_match:
                shortcut = data
            else:
                shortcut = self.layers.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride,
                                                   no_bias=True, workspace=workspace, name=name+'_sc_conv',)
                shortcut = self.layers.BatchNorm(data=shortcut, fix_gamma=False, eps=2e-5, momentum=bn_mom,
                                                 name=name + '_sc_bn', cudnn_off=cudnn_bn_off)
            if memonger:
                shortcut._set_attr(mirror_stage='True')
            # Branch 2
            #   Block 2A
            conv1 = self.layers.Convolution(data=data, num_filter=int(num_filter*0.25), kernel=(1, 1), stride=(1, 1),
                                            pad=(0, 0), no_bias=True, workspace=workspace, name=name + '_conv1')
            bn1 = self.layers.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1',
                                        cudnn_off=cudnn_bn_off, act_type=act)
            act1 = self.layers.Activation(data=bn1, act_type='relu', name=name + '_relu1') if not fuse_bn_relu else bn1
            #   Block 2B
            conv2 = self.layers.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(3, 3), stride=stride,
                                            pad=(1, 1), no_bias=True, workspace=workspace, name=name + '_conv2')
            bn2 = self.layers.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2',
                                        cudnn_off=cudnn_bn_off, act_type=act)
            act2 = self.layers.Activation(data=bn2, act_type='relu', name=name + '_relu2') if not fuse_bn_relu else bn2
            #   Block 3B
            conv3 = self.layers.Convolution(data=act2, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                                            no_bias=True, workspace=workspace, name=name + '_conv3')
            if fuse_bn_add_relu:
                return self.layers.BatchNormAddRelu(data=conv3, addend=shortcut, axis=1, fix_gamma=False, eps=2e-5,
                                                    momentum=bn_mom, cudnn_off=cudnn_bn_off)
            else:
                bn3 = self.layers.BatchNorm(data=conv3, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3',
                                            cudnn_off=cudnn_bn_off)
                # Element-wise summation and ReLU
                return self.layers.Activation(data=shortcut + bn3, act_type='relu', name=name + '_relu')
        else:
            # Branch 1
            if dim_match:
                shortcut = data
            else:
                shortcut = self.layers.Convolution(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride,
                                                   no_bias=True, workspace=workspace, name=name+'_sc_conv')
                shortcut = self.layers.BatchNorm(data=shortcut, fix_gamma=False, momentum=bn_mom, eps=2e-5,
                                                 name=name + '_sc_bn', cudnn_off=cudnn_bn_off)
            if memonger:
                shortcut._set_attr(mirror_stage='True')
            # Branch 2
            #   Block 2A
            conv1 = self.layers.Convolution(data=data, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                                            no_bias=True, workspace=workspace, name=name + '_conv1')
            bn1 = self.layers.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1',
                                        cudnn_off=cudnn_bn_off, act_type=act)
            act1 = self.layers.Activation(data=bn1, act_type='relu', name=name + '_relu1') if not fuse_bn_relu else bn1
            #   Block 2B
            conv2 = self.layers.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                                            no_bias=True, workspace=workspace, name=name + '_conv2')
            if fuse_bn_add_relu:
                return self.layers.BatchNormAddRelu(data=conv2, addend=shortcut, axis=1, fix_gamma=False, eps=2e-5,
                                                    momentum=bn_mom, cudnn_off=cudnn_bn_off)
            else:
                bn2 = self.layers.BatchNorm(data=conv2, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2',
                                            cudnn_off=cudnn_bn_off)
                # Element-wise summation and ReLU
                return self.layers.Activation(data=shortcut + bn2, act_type='relu', name=name + '_relu')

    def resnet(self, units, num_stages, filter_list, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False,
               cudnn_bn_off=False, fuse_bn_relu=False, fuse_bn_add_relu=False):
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
        """
        act = 'relu' if fuse_bn_relu else None
        num_unit = len(units)
        assert num_unit == num_stages
        v = self.add_data_node()
        v = Layers.conv_transform_layout(v, self.params['input_layout'], self.params['model_layout'])

        v = self.layers.Convolution(data=v, num_filter=filter_list[0], kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                                    no_bias=True, name="conv0", workspace=workspace)
        v = self.layers.BatchNorm(data=v, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0',
                                  cudnn_off=cudnn_bn_off, act_type=act)
        if not fuse_bn_relu:
            v = self.layers.Activation(data=v, act_type='relu', name='relu0')
        v = self.layers.Pooling(data=v, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max')

        for i in range(num_stages):
            v = self.residual_unit(v, filter_list[i+1], (1 if i == 0 else 2, 1 if i == 0 else 2),
                                   False, name='stage%d_unit%d' % (i + 1, 1),
                                   bottle_neck=bottle_neck, workspace=workspace, memonger=memonger,
                                   cudnn_bn_off=cudnn_bn_off, fuse_bn_relu=fuse_bn_relu,
                                   fuse_bn_add_relu=fuse_bn_add_relu)
            for j in range(units[i]-1):
                v = self.residual_unit(v, filter_list[i+1], (1, 1), True,
                                       name='stage%d_unit%d' % (i + 1, j + 2),
                                       bottle_neck=bottle_neck, workspace=workspace, memonger=memonger,
                                       cudnn_bn_off=cudnn_bn_off, fuse_bn_relu=fuse_bn_relu,
                                       fuse_bn_add_relu=fuse_bn_add_relu)

        # Although kernel is not used here when global_pool=True, we should put one
        v = self.layers.Pooling(data=v, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
        v = mx.sym.Flatten(data=v)
        v = self.add_head_nodes(v)
        return v

    def __init__(self, params):
        specs = ResNet.specs[params['model']]
        Model.check_parameters(
            params,
            {'name': specs['name'], 'num_classes': 1000, 'phase': 'training', 'dtype': 'float32',
             'input_layout': 'NCHW', 'model_layout': 'NCHW', 'nvidia_layers': False,
             'workspace': 1024}
        )
        params['input_shape'] = Model.conv_shape(3, (224, 224), params['input_layout'])

        Model.__init__(self, params)

        self.params = params
        self.layers = Layers(params)

        # Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
        # Original author Wei Wu
        # Some optimizations are taken from NVIDIA code from NGC containers.
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
            workspace=params['workspace'],
            fuse_bn_add_relu=params['nvidia_layers'],
            fuse_bn_relu=params['nvidia_layers']
        )
