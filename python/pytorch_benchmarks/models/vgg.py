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
import torch.nn as nn
from pytorch_benchmarks.models.model import Model

class VGG(Model):

    implements = ['vgg11', 'vgg13', 'vgg16', 'vgg19']

    specs = {
        'vgg11':{ 'name': 'VGG11', 'specs': ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]) },  # pylint: disable=C0326
        'vgg13':{ 'name': 'VGG13', 'specs': ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]) },  # pylint: disable=C0326
        'vgg16':{ 'name': 'VGG16', 'specs': ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]) },  # pylint: disable=C0326
        'vgg19':{ 'name': 'VGG19', 'specs': ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512]) }   # pylint: disable=C0326
    }

    def __init__(self, params):
        specs = VGG.specs[params['model']]
        Model.check_parameters(
            params,
            {'name': specs['name'], 'input_shape': (3, 224, 224),
             'num_classes': 1000, 'phase': 'training',
             'dtype': 'float32'}
        )
        Model.__init__(self, params)
        # Features (CNN)
        self.features = nn.Sequential()
        layers, filters = specs['specs']
        prev_filters = self.input_shape[0]
        for i, num in enumerate(layers):
            for j in range(num):
                self.features.add_module(
                    'conv%d_%d' % (i+1, j+1),
                    nn.Conv2d(prev_filters, filters[i], kernel_size=3, padding=1)
                )
                self.features.add_module('relu%d_%d' % (i+1, j+1), nn.ReLU(inplace=True))
                prev_filters = filters[i]
            self.features.add_module('pool%d' % (i+1), nn.MaxPool2d(kernel_size=2, stride=2))
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            nn.Linear(4096, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 512*7*7)
        x = self.classifier(x)
        return x
