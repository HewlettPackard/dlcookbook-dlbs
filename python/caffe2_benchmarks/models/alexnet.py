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
    https://github.com/caffe2/caffe2/blob/master/caffe2/python/helpers/conv.py
"""
from __future__ import absolute_import
from caffe2.python import brew
from caffe2_benchmarks.models.model import Model

class AlexNet(Model):
    """AlexNet neural network model."""
    
    implements = 'alexnet'

    def __init__(self, params):
        Model.check_parameters(
            params,
            {'name': 'AlexNet', 'input_shape':((3, 227, 227)),
             'num_classes': 1000, 'arg_scope': {'order': 'NCHW'}}
        )
        Model.__init__(self, params)

    def forward_pass_builder(self, model, loss_scale=1.0):
        """
            This function adds the operators, layers to the network. It should return a list
            of loss-blobs that are used for computing the loss gradient. This function is
            also passed an internally calculated loss_scale parameter that is used to scale
            your loss to normalize for the number of GPUs. Signature: function(model, loss_scale)
        """
        is_inference = self.phase == 'inference'

        v = 'data'

        v = brew.conv(model, v, 'conv1', 3, 96, kernel=11, stride=4)
        v = brew.relu(model, v, 'relu1')
        v = brew.lrn(model, v, 'lrn1', size=5, alpha=0.0001, beta=0.75)
        v = brew.max_pool(model, v, 'pool1', kernel=3, stride=2)

        v = brew.conv(model, v, 'conv2', 96, 256, kernel=5, pad=2, group=2)
        v = brew.relu(model, v, 'relu2')
        v = brew.lrn(model, v, 'lrn2', size=5, alpha=0.0001, beta=0.75)
        v = brew.max_pool(model, v, 'pool2', kernel=3, stride=2)

        v = brew.conv(model, v, 'conv3', 256, 384, kernel=3, pad=1)
        v = brew.relu(model, v, 'relu3')

        v = brew.conv(model, v, 'conv4', 384, 384, kernel=3, pad=1, group=2)
        v = brew.relu(model, v, 'relu4')

        v = brew.conv(model, v, 'conv5', 384, 256, kernel=3, pad=1, group=2)
        v = brew.relu(model, v, 'relu5')
        v = brew.max_pool(model, v, 'pool5', kernel=3, stride=2)

        v = brew.fc(model, v, 'fc6', dim_in=9216, dim_out=4096)
        v = brew.relu(model, v, 'relu6')
        v = brew.dropout(model, v, 'dropout6', ratio=0.5, is_test=is_inference)

        v = brew.fc(model, v, 'fc7', dim_in=4096, dim_out=4096)
        v = brew.relu(model, v, 'relu7')
        v = brew.dropout(model, v, 'dropout7', ratio=0.5, is_test=is_inference)

        return self.add_head_nodes(model, v, 4096, 'fc8', loss_scale=loss_scale)
