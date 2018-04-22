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
import mxnet as mx
from mxnet_benchmarks.models.model import Model

class AcousticModel(Model):
    
    implements = 'acoustic_model'
    
    @property
    def output(self):
        return self.__output

    def __init__(self, params):
        Model.check_parameters(
            params,
            {'name': 'AcousticModel', 'input_shape':(540),
             'num_classes': 8192, 'phase': 'training',
             'dtype': 'float32'}
        )
        Model.__init__(self, params)

        v = self.add_data_node()

        for _ in range(5):
            v = mx.sym.FullyConnected(data=v, num_hidden=2048)
            v = mx.symbol.Activation(data=v, act_type="relu")

        self.__output = self.add_head_nodes(v)
