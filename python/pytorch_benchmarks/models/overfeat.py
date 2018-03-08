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
import torch.nn as nn
from pytorch_benchmarks.models.model import Model

class Overfeat(Model):

    implements = 'overfeat'

    def __init__(self, params):
        Model.check_parameters(
            params,
            {'name': 'Overfeat', 'input_shape': (3, 231, 231), 'num_classes': 1000,
             'phase': 'training',
             'dtype': 'float32'}
        )
        Model.__init__(self, params)

        self.features = nn.Sequential(
            # Layer1
            nn.Conv2d(self.input_shape[0], 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer2
            nn.Conv2d(96, 256, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer3
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Layer4
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Layer5
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            # Layer6
            nn.Linear(1024 * 6 * 6, 3072),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # Layer7
            nn.Linear(3072, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(4096, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 1024 * 6 * 6)
        return self.classifier(x)
