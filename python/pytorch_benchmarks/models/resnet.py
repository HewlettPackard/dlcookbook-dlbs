# -*- coding: utf-8 -*-

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
import torch.nn as nn
from pytorch_benchmarks.models.model import Model


class ResnetModule(nn.Module):
    """Number of outut channels is always 'num_filters'."""

    def __init__(self, num_input_channels, num_filters, stride, dim_match, bottle_neck):
        """Number of outut channels is always 'num_filters'."""
        super(ResnetModule, self).__init__()
        # Branch 1
        if dim_match:
            self.shortcut = None
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(num_input_channels, num_filters, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_filters, eps=2e-5, momentum=0.9, affine=True)
            )
        # Branch 2
        if bottle_neck:
            bottleneck_channels = num_filters/4
            self.main = nn.Sequential(
                # Block 2A
                nn.Conv2d(num_input_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(bottleneck_channels, eps=2e-5, momentum=0.9, affine=True),
                nn.ReLU(inplace=True),
                # Block 2B
                nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(bottleneck_channels, eps=2e-5, momentum=0.9, affine=True),
                nn.ReLU(inplace=True),
                # Block 3B
                nn.Conv2d(bottleneck_channels, num_filters, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_filters, eps=2e-5, momentum=0.9, affine=True),
            )
        else:
            self.main = nn.Sequential(
                # Block 2A
                nn.Conv2d(num_input_channels, num_filters, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(num_filters, eps=2e-5, momentum=0.9, affine=True),
                nn.ReLU(inplace=True),
                # Block 2B
                nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_filters, eps=2e-5, momentum=0.9, affine=True),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x if self.shortcut is None else self.shortcut(x)
        return self.relu(shortcut + self.main(x))


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

        self.features = nn.Sequential(
            nn.Conv2d(3, filter_list[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(filter_list[0], eps=2e-5, momentum=0.9, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Number of stages is always 4
        num_prev_channels = filter_list[0]
        for i in range(len(specs['units'])):
            # num_input_channels, num_filters, stride, dim_match, bottle_neck
            self.features.add_module(
                'stage%d_unit%d' % (i + 1, 1),
                ResnetModule(num_prev_channels, filter_list[i+1], (1 if i == 0 else 2), False, bottle_neck)
            )
            num_prev_channels = filter_list[i+1]
            for j in range(specs['units'][i]-1):
                self.features.add_module(
                    'stage%d_unit%d' % (i + 1, j + 2),
                    ResnetModule(num_prev_channels, filter_list[i+1], 1, True, bottle_neck)
                )
        self.features.add_module('pool1', nn.AvgPool2d(kernel_size=7, padding=0))
        self.num_output_channels = filter_list[-1]

        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_channels, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.num_output_channels)
        x = self.classifier(x)
        return x
