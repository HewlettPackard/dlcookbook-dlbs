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
    https://caffe2.ai/docs/SynchronousSGD.html
"""
from __future__ import absolute_import
from caffe2.python import brew
from caffe2_benchmarks.models.model import Model

class AcousticModel(Model):
    """Some deep FCNN used in hybrid HMM/DNN speech recognition systems."""
    
    implements = 'acoustic_model'

    def __init__(self, params):
        Model.check_parameters(
            params,
            {'name':'AcousticModel', 'input_shape':(540),
             'num_classes': 8192, 'arg_scope': {'order': 'NCHW'}}
        )
        Model.__init__(self, params)

    def forward_pass_builder(self, model, loss_scale=1.0):
        """
            This function adds the operators, layers to the network. It should return
            a list of loss-blobs that are used for computing the loss gradient. This
            function is also passed an internally calculated loss_scale parameter that
            is used to scale your loss to normalize for the number of GPUs.
            Signature: function(model, loss_scale)
        """
        v = 'data'
        dim_in = self.input_shape[0]
        for idx in range(5):
            v = brew.fc(model, v, 'fc%d' % (idx+1), dim_in=dim_in, dim_out=2048)
            v = brew.relu(model, v, 'relu%d' % (idx+1))
            dim_in = 2048

        return self.add_head_nodes(model, v, dim_in, 'fc%d' % (idx+2), loss_scale=loss_scale)
