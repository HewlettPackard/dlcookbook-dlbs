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

class VGG(Model):
    
    implements = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
    
    specs = {
        'vgg11':{ 'name': 'VGG11', 'specs': ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]) },  # pylint: disable=C0326
        'vgg13':{ 'name': 'VGG13', 'specs': ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]) },  # pylint: disable=C0326
        'vgg16':{ 'name': 'VGG16', 'specs': ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]) },  # pylint: disable=C0326
        'vgg19':{ 'name': 'VGG19', 'specs': ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512]) }   # pylint: disable=C0326
    }
    @property
    def output(self):
        return self.__output

    def __init__(self, params):
        specs = VGG.specs[params['model']]
        Model.check_parameters(
            params,
            {'name': specs['name'], 'input_shape': (3, 224, 224),
             'num_classes': 1000, 'phase': 'training',
             'dtype': 'float32'}
        )
        Model.__init__(self, params)
        training = self.phase == 'training'

        v = mx.sym.Variable(name="data")

        layers, filters = specs['specs']
        for i, num in enumerate(layers):
            for j in range(num):
                v = mx.symbol.Convolution(name='conv%d_%d' % (i+1, j+1), data=v, kernel=(3, 3), pad=(1, 1), num_filter=filters[i])
                v = mx.symbol.Activation(name='relu%d_%d' % (i+1, j+1), data=v, act_type="relu")
            v = mx.sym.Pooling(name='pool%d' % (i+1), data=v, pool_type="max", kernel=(2, 2), stride=(2, 2))

        v = mx.sym.Flatten(name='flatten', data=v)

        for i in range(2):
            v = mx.sym.FullyConnected(name='fc%d' % (6+i), data=v, num_hidden=4096)
            v = mx.symbol.Activation(name='relu%d' % (6+i), data=v, act_type="relu")
            v = mx.symbol.Dropout(name='drop%d' % (6+i), data=v, p=0.5) if training else v

        self.__output = self.add_head_nodes(v)
