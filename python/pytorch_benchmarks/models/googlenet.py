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
from __future__ import absolute_import
import torch
import torch.nn as nn
from pytorch_benchmarks.models.model import Model

class ConvModule(nn.Module):
    """
        [input] -> Conv2D -> ReLU -> [output]
    """
    def __init__(self, num_input_channels, num_filters, kernel_size,
                 stride=1, padding=0):
        super(ConvModule, self).__init__()
        self.conv_module = nn.Sequential(
            nn.Conv2d(num_input_channels, num_filters, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_module(x)


class InceptionModule(nn.Module):
    """
                 |            -> c1x1  |                         branch: conv1
                 | -> c3x3r   -> c3x3  |                         branch: conv3
        [input] -|                     |-> concat -> [output]
                 | -> cd5x5r  -> cd5x5 |                         branch: conv5
                 | -> pooling -> cproj |                         branch: pooling
    """
    def __init__(self, num_input_channels, num_1x1, num_3x3red, num_3x3, num_d5x5red,
                 num_d5x5, proj):
        super(InceptionModule, self).__init__()
        self.conv1 = ConvModule(num_input_channels, num_filters=num_1x1, kernel_size=1)
        self.conv3 = nn.Sequential(
            ConvModule(num_input_channels, num_filters=num_3x3red, kernel_size=1),
            ConvModule(num_3x3red, num_filters=num_3x3, kernel_size=3, padding=1)
        )
        self.conv5 = nn.Sequential(
            ConvModule(num_input_channels, num_filters=num_d5x5red, kernel_size=1),
            ConvModule(num_d5x5red, num_filters=num_d5x5, kernel_size=5, padding=2)
        )
        self.pooling = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvModule(num_input_channels, num_filters=proj, kernel_size=1)
        )


    def forward(self, x):
        return torch.cat(
            [self.conv1(x), self.conv3(x), self.conv5(x), self.pooling(x)],
            dim=1
        )


class GoogleNet(Model):

    implements = 'googlenet'

    def __init__(self, params):
        """"""
        Model.check_parameters(
            params,
            {'name': 'GoogleNet', 'input_shape':(3, 224, 224),
             'num_classes': 1000, 'phase': 'training',
             'dtype': 'float32'}
        )
        Model.__init__(self, params)
        self.features = nn.Sequential(
            ConvModule(self.input_shape[0], 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),

            ConvModule(64, 64, kernel_size=1, stride=1),

            ConvModule(64, 192, kernel_size=3, stride=1, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            InceptionModule(192, num_1x1=64, num_3x3red=96, num_3x3=128, num_d5x5red=16, num_d5x5=32, proj=32),  # out channels = 256
            InceptionModule(256, num_1x1=128, num_3x3red=128, num_3x3=192, num_d5x5red=32, num_d5x5=96, proj=64),  # out channels = 480
            nn.MaxPool2d(kernel_size=3, stride=2),

            InceptionModule(480, num_1x1=192, num_3x3red=96, num_3x3=208, num_d5x5red=16, num_d5x5=48, proj=64),  # out channels = 512
            InceptionModule(512, num_1x1=160, num_3x3red=112, num_3x3=224, num_d5x5red=24, num_d5x5=64, proj=64),  # out channels = 512
            InceptionModule(512, num_1x1=128, num_3x3red=128, num_3x3=256, num_d5x5red=24, num_d5x5=64, proj=64),  # out channels = 512
            InceptionModule(512, num_1x1=112, num_3x3red=144, num_3x3=288, num_d5x5red=32, num_d5x5=64, proj=64),  # out channels = 528
            InceptionModule(528, num_1x1=256, num_3x3red=160, num_3x3=320, num_d5x5red=32, num_d5x5=128, proj=128), # out channels = 832
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            InceptionModule(832, num_1x1=256, num_3x3red=160, num_3x3=320, num_d5x5red=32, num_d5x5=128, proj=128), # out channels = 832
            InceptionModule(832, num_1x1=384, num_3x3red=192, num_3x3=384, num_d5x5red=48, num_d5x5=128, proj=128), # out channels = 1024
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 1024 * 1 * 1)
        return self.classifier(x)
