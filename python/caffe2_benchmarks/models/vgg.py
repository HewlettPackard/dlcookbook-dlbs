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
"""Differnt VGG models implementation.

The models are: VGG11, VGG13, VGG16 and VGG19
"""
from caffe2.python import brew
from caffe2_benchmarks.models.model import Model

class VGG(Model):
    """Differnt VGG models implementation."""
    
    implements = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
    
    specs = {
        'vgg11':{ 'name': 'VGG11', 'specs': ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]) }, # pylint: disable=C0326
        'vgg13':{ 'name': 'VGG13', 'specs': ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]) }, # pylint: disable=C0326
        'vgg16':{ 'name': 'VGG16', 'specs': ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]) }, # pylint: disable=C0326
        'vgg19':{ 'name': 'VGG19', 'specs': ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512]) }  # pylint: disable=C0326
    }

    def __init__(self, params):
        specs = VGG.specs[params['model']]
        Model.check_parameters(
            params,
            {'name': specs['name'], 'input_shape':(3, 224, 224),
             'num_classes': 1000, 'arg_scope': {'order': 'NCHW'}}
        )
        Model.__init__(self, params)
        self.__model = params['model']

    def forward_pass_builder(self, model, loss_scale=1.0):
        """
            This function adds the operators, layers to the network. It should return
            a list of loss-blobs that are used for computing the loss gradient. This
            function is also passed an internally calculated loss_scale parameter that
            is used to scale your loss to normalize for the number of GPUs.
            Signature: function(model, loss_scale)
        """
        is_inference = self.phase == 'inference'
        layers, filters = VGG.specs[self.__model]['specs']
        v = 'data'
        dim_in = self.input_shape[0]
        for i, num in enumerate(layers):
            for j in range(num):
                v = brew.conv(model, v, 'conv%d_%d' % (i+1, j+1), dim_in, filters[i], kernel=3, pad=1)
                v = brew.relu(model, v, 'relu%d_%d' % (i+1, j+1))
                dim_in = filters[i]
            v = brew.max_pool(model, v, 'pool%d' % (i+1), kernel=2, stride=2)

        dim_in = 25088 # 512 * 7 * 7 (output tensor of previous max pool layer)
        for i in range(2):
            v = brew.fc(model, v, 'fc%d' % (6+i), dim_in=dim_in, dim_out=4096)
            v = brew.relu(model, v, 'relu%d' % (6+i))
            v = brew.dropout(model, v, 'drop%d' % (6+i), ratio=0.5, is_test=is_inference)
            dim_in = 4096

        return self.add_head_nodes(model, v, 4096, 'fc8', loss_scale=loss_scale)
