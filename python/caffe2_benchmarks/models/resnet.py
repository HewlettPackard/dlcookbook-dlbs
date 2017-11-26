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
from caffe2.python import brew
from caffe2_benchmarks.models.model import Model

class ResNet(Model):

    implements = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101',
        'resnet152', 'resnet200', 'resnet269'
    ]

    specs = {
        'resnet18':  { 'name': 'ResNet18',  'units': [2, 2,  2,  2], 'num_layers': 18  },  # pylint: disable=C0326
        'resnet34':  { 'name': 'ResNet34',  'units': [3, 4,  6,  3], 'num_layers': 34  },  # pylint: disable=C0326
        'resnet50':  { 'name': 'ResNet50',  'units': [3, 4,  6,  3], 'num_layers': 50  },  # pylint: disable=C0326
        'resnet101': { 'name': 'ResNet101', 'units': [3, 4,  23, 3], 'num_layers': 101 },  # pylint: disable=C0326
        'resnet152': { 'name': 'ResNet152', 'units': [3, 8,  36, 3], 'num_layers': 152 },  # pylint: disable=C0326
        'resnet200': { 'name': 'ResNet200', 'units': [3, 24, 36, 3], 'num_layers': 200 },  # pylint: disable=C0326
        'resnet269': { 'name': 'ResNet269', 'units': [3, 30, 48, 8], 'num_layers': 269 }   # pylint: disable=C0326
    }

    def residual_unit(self, model, v, num_in_channels, num_filter, stride, dim_match,
                      name, bottle_neck=True, bn_mom=0.9, is_inference=False):
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
                shortcut = v
            else:
                shortcut = brew.conv(model, v, name+'_sc', num_in_channels, num_filter,
                                     kernel=1, stride=stride, no_bias=True)
                shortcut = brew.spatial_bn(model, shortcut, name+'_sc_bn', num_filter,
                                           eps=2e-5, momentum=bn_mom, is_test=is_inference)
            # Branch 2
            interim_filters = int(num_filter*0.25)
            #     Block 1
            conv1 = brew.conv(model, v, name+'_conv1', num_in_channels, interim_filters,
                              kernel=1, pad=0, stride=1, no_bias=True)
            bn1 = brew.spatial_bn(model, conv1, name+'_bn1', interim_filters, eps=2e-5,
                                  momentum=bn_mom, is_test=is_inference)
            act1 = brew.relu(model, bn1, name+'_relu1')
            #     Block 2
            conv2 = brew.conv(model, act1, name+'_conv2', interim_filters,
                              int(num_filter*0.25), kernel=3, pad=1, stride=stride, no_bias=True)
            bn2 = brew.spatial_bn(model, conv2, name+'_bn2', interim_filters,
                                  eps=2e-5, momentum=bn_mom, is_test=is_inference)
            act2 = brew.relu(model, bn2, name+'_relu2')
            #     Block 3
            conv3 = brew.conv(model, act2, name+'_conv3', interim_filters, num_filter,
                              kernel=1, pad=0, stride=1, no_bias=True)
            bn3 = brew.spatial_bn(model, conv3, name+'_bn3', num_filter, eps=2e-5,
                                  momentum=bn_mom, is_test=is_inference)
            # Element-wise summation and ReLU
            output = brew.sum(model, [shortcut, bn3], name+'_sum')
            output = brew.relu(model, output, name+'_relu')
            return output
        else:
            # Branch 1
            if dim_match:
                shortcut = v
            else:
                shortcut = brew.conv(model, v, name+'_sc_conv', num_in_channels, num_filter,
                                     kernel=1, stride=stride, no_bias=True)
                shortcut = brew.spatial_bn(model, shortcut, name+'_sc_bn', num_filter,
                                           eps=2e-5, momentum=bn_mom, is_test=is_inference)
            # Branch 2
            #     Block 1
            conv1 = brew.conv(model, v, name+'_conv1', num_in_channels, num_filter, kernel=3,
                              pad=1, stride=stride, no_bias=True)
            bn1 = brew.spatial_bn(model, conv1, name+'_bn1', num_filter, eps=2e-5,
                                  momentum=bn_mom, is_test=is_inference)
            act1 = brew.relu(model, bn1, name+'_relu1')
            #     Block 2
            conv2 = brew.conv(model, act1, name+'_conv2', num_filter, num_filter,
                              kernel=3, pad=1, stride=1, no_bias=True)
            bn2 = brew.spatial_bn(model, conv2, name+'_bn2', num_filter, eps=2e-5,
                                  momentum=bn_mom, is_test=is_inference)
            # Element-wise summation and ReLU
            output = brew.sum(model, [shortcut, bn2], name+'_sum')
            output = brew.relu(model, output, name+'_relu')
            return output

    def resnet(self, model, units, num_stages, filter_list, bottle_neck=True, bn_mom=0.9,
               is_inference=False, loss_scale=1.0):
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
            Workspace used in convolution operator''
        dtype : str
            Precision (float32 or float16)
        """
        num_unit = len(units)
        assert num_unit == num_stages
        v = 'data'
        (nchannel, _, _) = self.input_shape # (nchannel, height, width)

        v = brew.conv(model, v, 'conv0', nchannel, filter_list[0], kernel=7, pad=3,
                      stride=2, no_bias=True)
        v = brew.spatial_bn(model, v, 'bn0', filter_list[0], eps=2e-5, momentum=bn_mom,
                            is_test=is_inference)
        v = brew.relu(model, v, 'relu0')
        v = brew.max_pool(model, v, 'pool0', kernel=3, stride=2, pad=1)

        dim_in = filter_list[0]
        for i in range(num_stages):
            v = self.residual_unit(model, v, dim_in, filter_list[i+1], stride=(1 if i == 0 else 2),
                                   dim_match=False,
                                   name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck,
                                   is_inference=is_inference)
            dim_in = filter_list[i+1]
            for j in range(units[i]-1):
                v = self.residual_unit(model, v, dim_in, filter_list[i+1], 1, True,
                                       name='stage%d_unit%d' % (i + 1, j + 2),
                                       bottle_neck=bottle_neck, is_inference=is_inference)

        v = brew.average_pool(model, v, 'pool1', kernel=7, global_pool=True)
        return self.add_head_nodes(model, v, dim_in, 'classifier', loss_scale=loss_scale)

    def __init__(self, params):
        specs = ResNet.specs[params['model']]
        Model.check_parameters(
            params,
            {'name': specs['name'], 'input_shape':(3, 224, 224),
             'num_classes': 1000, 'arg_scope': {'order': 'NCHW'}}
        )
        Model.__init__(self, params)
        self.__model = params['model']

    def forward_pass_builder(self, model, loss_scale=1.0):
        """
        Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
        Original author Wei Wu
        """
        specs = ResNet.specs[self.__model]
        if specs['num_layers'] >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False

        return self.resnet(
            model,
            units=specs['units'],
            num_stages=4,
            filter_list=filter_list,
            bottle_neck=bottle_neck,
            is_inference=(self.phase == 'inference'),
            loss_scale=loss_scale
        )


# Original (buggy) implementation not consistent with reference Caffe's descriptors
#class ResNet(Model):
#    
#    implements = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200', 'resnet269']
#    
#    specs = {
#        'resnet18':  { 'name': 'ResNet18',  'units': [2, 2,  2,  2], 'num_layers': 18  },  # pylint: disable=C0326
#        'resnet34':  { 'name': 'ResNet34',  'units': [3, 4,  6,  3], 'num_layers': 34  },  # pylint: disable=C0326
#        'resnet50':  { 'name': 'ResNet50',  'units': [3, 4,  6,  3], 'num_layers': 50  },  # pylint: disable=C0326
#        'resnet101': { 'name': 'ResNet101', 'units': [3, 4,  23, 3], 'num_layers': 101 },  # pylint: disable=C0326
#        'resnet152': { 'name': 'ResNet152', 'units': [3, 8,  36, 3], 'num_layers': 152 },  # pylint: disable=C0326
#        'resnet200': { 'name': 'ResNet200', 'units': [3, 24, 36, 3], 'num_layers': 200 },  # pylint: disable=C0326
#        'resnet269': { 'name': 'ResNet269', 'units': [3, 30, 48, 8], 'num_layers': 269 }   # pylint: disable=C0326
#    }
#
#    def residual_unit(self, model, v, num_in_channels, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, is_inference=False):
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
#            bn1 = brew.spatial_bn(model, v, name+'_bn1', num_in_channels, eps=2e-5, momentum=bn_mom, is_test=is_inference)
#            act1 = brew.relu(model, bn1, name+'_relu1')
#            conv1 = brew.conv(model, act1, name+'_conv1', num_in_channels, int(num_filter*0.25), kernel=1, pad=0, stride=1, no_bias=True)
#
#
#            bn2 = brew.spatial_bn(model, conv1, name+'_bn2', int(num_filter*0.25), eps=2e-5, momentum=bn_mom, is_test=is_inference)
#            act2 = brew.relu(model, bn2, name+'_relu2')
#            conv2 = brew.conv(model, act2, name+'_conv2', int(num_filter*0.25), int(num_filter*0.25), kernel=3, pad=1, stride=stride, no_bias=True)
#
#            bn3 = brew.spatial_bn(model, conv2, name+'_bn3', int(num_filter*0.25), eps=2e-5, momentum=bn_mom, is_test=is_inference)
#            act3 = brew.relu(model, bn3, name+'_relu3')
#            conv3 = brew.conv(model, act3, name+'_conv3', int(num_filter*0.25), num_filter, kernel=1, pad=0, stride=1, no_bias=True)
#
#            if dim_match:
#                shortcut = v
#            else:
#                shortcut = brew.conv(model, act1, name+'_sc', num_in_channels, num_filter, kernel=1, stride=stride, no_bias=True)
#
#            return brew.sum(model, [conv3, shortcut], name+'_concat')
#        else:
#            bn1 = brew.spatial_bn(model, v, name+'_bn1', num_in_channels, eps=2e-5, momentum=bn_mom, is_test=is_inference)
#            act1 = brew.relu(model, bn1, name+'_relu1')
#            conv1 = brew.conv(model, act1, name+'_conv1', num_in_channels, num_filter, kernel=3, pad=1, stride=stride, no_bias=True)
#
#            bn2 = brew.spatial_bn(model, conv1, name+'_bn2', num_filter, eps=2e-5, momentum=bn_mom, is_test=is_inference)
#            act2 = brew.relu(model, bn2, name+'_relu2')
#            conv2 = brew.conv(model, act2, name+'_conv2', num_filter, num_filter, kernel=3, pad=1, stride=1, no_bias=True)
#
#            if dim_match:
#                shortcut = v
#            else:
#                shortcut = brew.conv(model, act1, name+'_sc', num_in_channels, num_filter, kernel=1, stride=stride, no_bias=True)
#
#            return brew.sum(model, [conv2, shortcut], name+'_concat')
#
#    def resnet(self, model, units, num_stages, filter_list, bottle_neck=True, bn_mom=0.9, is_inference=False, loss_scale=1.0):
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
#            Workspace used in convolution operator''
#        dtype : str
#            Precision (float32 or float16)
#        """
#        num_unit = len(units)
#        assert num_unit == num_stages
#        v = 'data'
#        (nchannel, height, _) = self.input_shape # (nchannel, height, width)
#        v = brew.spatial_bn(model, v, 'bn_data', nchannel, eps=2e-5, momentum=bn_mom, is_test=is_inference)
#        if height <= 32:            # such as cifar10
#            v = brew.conv(model, v, 'conv0', nchannel, filter_list[0], kernel=3, pad=1, stride=1, no_bias=True)
#        else:                       # often expected to be 224 such as imagenet
#            v = brew.conv(model, v, 'conv0', nchannel, filter_list[0], kernel=7, pad=3, stride=2, no_bias=True)
#            v = brew.spatial_bn(model, v, 'bn0', filter_list[0], eps=2e-5, momentum=bn_mom, is_test=is_inference)
#            v = brew.relu(model, v, 'relu0')
#            v = brew.max_pool(model, v, 'pool0', kernel=3, stride=2, pad=1)
#
#        # def residual_unit(self, model, v, num_in_channels, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, is_inference=False):
#        dim_in = filter_list[0]
#        for i in range(num_stages):
#            v = self.residual_unit(model, v, dim_in, filter_list[i+1], stride=(1 if i == 0 else 2), dim_match=False,
#                                   name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, is_inference=is_inference)
#            dim_in = filter_list[i+1]
#            for j in range(units[i]-1):
#                v = self.residual_unit(model, v, dim_in, filter_list[i+1], 1, True, name='stage%d_unit%d' % (i + 1, j + 2),
#                                       bottle_neck=bottle_neck, is_inference=is_inference)
#        v = brew.spatial_bn(model, v, 'bn1', dim_in, eps=2e-5, momentum=bn_mom, is_test=is_inference)
#        v = brew.relu(model, v, 'relu1')
#        # Although kernel is not used here when global_pool=True, we should put one
#        v = brew.average_pool(model, v, 'pool1', kernel=7, global_pool=True)
#
#        return self.add_head_nodes(model, v, dim_in, 'classifier', loss_scale=loss_scale)
#
#    def __init__(self, params):
#        specs = ResNet.specs[params['model']]
#        Model.check_parameters(
#            params,
#            {'name': specs['name'], 'input_shape':(3, 224, 224),
#             'num_classes': 1000, 'arg_scope': {'order': 'NCHW'}}
#        )
#        Model.__init__(self, params)
#        self.__model = params['model']
#
#    def forward_pass_builder(self, model, loss_scale=1.0):
#        """
#        Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
#        Original author Wei Wu
#        """
#        height = self.input_shape[1] #(nchannel, height, width)
#        specs = ResNet.specs[self.__model]
#        num_layers = specs['num_layers']
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
#        # def resnet(self, model, units, num_stages, filter_list, bottle_neck=True, bn_mom=0.9, is_inference=False, loss_scale=1.0):
#        return self.resnet(
#            model,
#            units=units,
#            num_stages=num_stages,
#            filter_list=filter_list,
#            bottle_neck=bottle_neck,
#            is_inference=(self.phase == 'inference'),
#            loss_scale=loss_scale
#        )