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
from caffe2.python import brew
from caffe2_benchmarks.models.model import Model

# https://github.com/dmlc/mxnet/blob/master/example/image-classification/symbols/googlenet.py
# http://ethereon.github.io/netscope/#/gist/4325909f3683e51eaf93fdaeed6b2a9b

def conv_factory(model, v, num_in_channels, num_filter, kernel, stride=1, pad=0, name=None, suffix=''):
    v = brew.conv(model, v, 'conv_%s%s' %(name, suffix), num_in_channels, num_filter, kernel=kernel, pad=pad, stride=stride)
    v = brew.relu(model, v, 'relu_%s%s' %(name, suffix))
    return v

def inception_factory(model, v, num_in_channels, num_1x1, num_3x3red, num_3x3, num_d5x5red, num_d5x5, proj, name):
    # 1x1
    c1x1 = conv_factory(model, v, num_in_channels, num_filter=num_1x1, kernel=1, name=('%s_1x1' % name))
    # 3x3 reduce + 3x3
    c3x3r = conv_factory(model, v, num_in_channels, num_filter=num_3x3red, kernel=1, name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = conv_factory(model, c3x3r, num_3x3red, num_filter=num_3x3, kernel=3, pad=1, name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd5x5r = conv_factory(model, v, num_in_channels, num_filter=num_d5x5red, kernel=1, name=('%s_5x5' % name), suffix='_reduce')
    cd5x5 = conv_factory(model, cd5x5r, num_d5x5red, num_filter=num_d5x5, kernel=5, pad=2, name=('%s_5x5' % name))
    # pool + proj
    pooling = brew.max_pool(model, v, 'max_pool_%s_pool' % name, kernel=3, stride=1, pad=1)
    cproj = conv_factory(model, pooling, num_in_channels, num_filter=proj, kernel=1, name=('%s_proj' %  name))
    # concat and return
    return brew.concat(model, [c1x1, c3x3, cd5x5, cproj], 'ch_concat_%s_chconcat' % name)

class GoogleNet(Model):
    """A GoogleNet implementation"""
    
    implements = 'googlenet'

    def __init__(self, params):
        """ Naming and topology according to: http://ethereon.github.io/netscope/#/gist/f2e4825a8d4f8a3609cefd7ffadc910a
            Based on: https://github.com/dmlc/mxnet/blob/master/example/image-classification/symbols/alexnet.py
        """
        Model.check_parameters(
            params,
            {'name': 'GoogleNet', 'input_shape':(3, 224, 224),
             'num_classes': 1000, 'arg_scope': {'order': 'NCHW'}})
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

        v = conv_factory(model, v, self.input_shape[0], 64, kernel=7, stride=2, pad=3, name="conv1/7x7_s2")
        v = brew.max_pool(model, v, 'pool1/3x3_s2', kernel=3, stride=2)
        v = brew.lrn(model, v, 'pool1/norm1', size=5, alpha=0.0001, beta=0.75)

        v = conv_factory(model, v, 64, 64, kernel=1, stride=1, name="conv2/3x3_reduce")

        v = conv_factory(model, v, 64, 192, kernel=3, stride=1, pad=1, name="conv2/3x3")
        v = brew.lrn(model, v, 'conv2/norm2', size=5, alpha=0.0001, beta=0.75)
        v = brew.max_pool(model, v, 'pool2/3x3_s2', kernel=3, stride=2)

        v = inception_factory(model, v, 192, 64,  96,  128, 16, 32, 32, name="inception_3a")
        v = inception_factory(model, v, 256, 128, 128, 192, 32, 96, 64, name="inception_3b")
        v = brew.max_pool(model, v, 'pool3/3x3_s2', kernel=3, stride=2)

        v = inception_factory(model, v, 480, 192, 96,  208, 16, 48,  64,  name="inception_4a")
        v = inception_factory(model, v, 512, 160, 112, 224, 24, 64,  64,  name="inception_4b")
        v = inception_factory(model, v, 512, 128, 128, 256, 24, 64,  64,  name="inception_4c")
        v = inception_factory(model, v, 512, 112, 144, 288, 32, 64,  64,  name="inception_4d")
        v = inception_factory(model, v, 528, 256, 160, 320, 32, 128, 128, name="inception_4e")
        v = brew.max_pool(model, v, 'pool4/3x3_s2', kernel=3, stride=2, pad=1)

        v = inception_factory(model, v, 832, 256, 160, 320, 32, 128, 128, name="inception_5a")
        v = inception_factory(model, v, 832, 384, 192, 384, 48, 128, 128, name="inception_5b")
        v = brew.average_pool(model, v, 'pool5/7x7_s1', kernel=7, stride=1)
        v = brew.dropout(model, v, 'pool5/drop_7x7_s1', ratio=0.5, is_test=(self.phase == 'inference'))

        return self.add_head_nodes(model, v, 1024, 'classifier', loss_scale=loss_scale)
