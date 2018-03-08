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

class SensorNet(Model):

    implements = 'sensor_net'

    def __init__(self, params):
        Model.check_parameters(
            params,
            {'name': 'SensorNet', 'input_shape':(784),
             'num_classes': 16, 'phase': 'training',
             'dtype': 'float32'}
        )
        Model.__init__(self, params)
        self.model = nn.Sequential()

        prev_size = self.input_shape[0]
        for idx, layer_size in enumerate([1024, 1024, 1024]):
            self.model.add_module('linear_%d' % idx, nn.Linear(prev_size, layer_size))
            self.model.add_module('relu_%d' % idx, nn.ReLU(inplace=True))
            prev_size = layer_size
        self.model.add_module('classifier', nn.Linear(prev_size, self.num_classes))

    def forward(self, x):
        return self.model(x)
