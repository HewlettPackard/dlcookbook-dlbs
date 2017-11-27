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
""" Based on tf_cnn_benchmarks implementation
    InceptionV3:
      http://ethereon.github.io/netscope/#/gist/04a797f778a7d513a9b52af4c1dbee4e
      https://4.bp.blogspot.com/-TMOLlkJBxms/Vt3HQXpE2cI/AAAAAAAAA8E/7X7XRFOY6Xo/s1600/image03.png
    InceptionV4:
      http://ethereon.github.io/netscope/#/gist/8fdab7a3ea5bceb9169832dfd73b5e31
"""
from __future__ import absolute_import
import mxnet as mx
from collections import defaultdict
from mxnet_benchmarks.models.model import Model

class BaseInceptionModel(Model):

    def conv(self, name, inputs, num_filters, kernel, stride, pad):
        self.counts[name] += 1
        name = name + str(self.counts[name]-1)
        v = mx.symbol.Convolution(name=name+"_conv", data=inputs, num_filter=num_filters,
                                  kernel=kernel, stride=stride, pad=pad, no_bias=True)
        v = mx.sym.BatchNorm(name=name+"_bn", data=v, fix_gamma=False,
                             eps=2e-5, momentum=0.9)
        v = mx.symbol.Activation(name=name+'_relu', data=v, act_type="relu")
        return v

    def inception_module(self, name, inputs, branches):
        """Add parallel branches from 'branhes' into current graph"""
        def get_tuple(value):
            return value if isinstance(value,tuple) else (value, value)
        self.counts[name] += 1
        name = name + str(self.counts[name]-1)
        layers_outputs = []             # Outputs of each layer in each branch
        for branch_id, branch in enumerate(branches):
            v = inputs                  # Inputs for a next layer in a branch
            layers_outputs.append([])   # Outputs of layers in this branch
            for layer_id, layer in enumerate(branch):
                if layer[0] == 'conv':
                    v = self.conv(
                        name='%s_b%d_l%d_' % (name, branch_id, layer_id), inputs=v,
                        num_filters=layer[1],
                        kernel=get_tuple(layer[2]), stride=get_tuple(layer[3]),
                        pad=get_tuple(layer[4])
                    )
                elif layer[0] == 'avg' or layer[0] == 'max':
                    v = mx.symbol.Pooling(
                        name='%s_b%d_p%d' % (name, branch_id, layer_id),
                        data=v, pool_type=layer[0], kernel=get_tuple(layer[1]),
                        stride=get_tuple(layer[2]), pad=get_tuple(layer[3])
                    )
                elif layer[0] == 'share':
                    v = layers_outputs[-2][layer_id]
                else:
                    assert False, 'Unknown later type - ' + layer[0]
                layers_outputs[-1].append(v)
        # concat
        concat = mx.symbol.Concat(
            *[outputs[-1] for outputs in layers_outputs],
            name='%s_concat' % name
        )
        return concat

    def __init__(self, params):
        Model.check_parameters(
            params,
            {'input_shape':(3, 299, 299), 'num_classes': 1000,
             'phase': 'training',
             'dtype': 'float32'}
        )
        Model.__init__(self, params)
        self.counts = defaultdict(lambda: 0)



class Inception3(BaseInceptionModel):

    implements = 'inception3'

    @property
    def output(self):
        return self.__output

    def module_a(self, inputs, n):
        branhes = [
            [('conv', 64, 1, 1, 0)],
            [('conv', 48, 1, 1, 0), ('conv', 64, 5, 1, 2)],
            [('conv', 64, 1, 1, 0), ('conv', 96, 3, 1, 1), ('conv', 96, 3, 1, 1)],
            [('avg', 3, 1, 1), ('conv', n, 1, 1, 0)]
        ]
        return self.inception_module('inception_a', inputs, branhes)

    def module_b(self, inputs):
        branches = [
            [('conv', 384, 3, 2, 0)],
            [('conv', 64, 1, 1, 0), ('conv', 96, 3, 1, 1), ('conv', 96, 3, 2, 0)],
            [('max', 3, 2, 0)]
        ]
        return self.inception_module('inception_b', inputs, branches)

    def module_c(self, inputs, n):
        branches = [
            [('conv', 192, 1, 1, 0)],
            [('conv', n, 1, 1, 0), ('conv', n, (1,7), 1, (0,3)), ('conv', 192, (7,1), 1, (3,0))],
            [('conv', n, 1, 1, 0), ('conv', n, (7,1), 1, (3,0)), ('conv', n, (1,7), 1, (0,3)),
             ('conv', n, (7,1), 1, (3,0)), ('conv', 192, (1,7), 1, (0,3))],
            [('avg', 3, 1, 1), ('conv', 192, 1, 1, 0)]
        ]
        return self.inception_module('inception_c', inputs, branches)

    def module_d(self, inputs):
        branches = [
            [('conv', 192, 1, 1, 0), ('conv', 320, 3, 2, 0)],
            [('conv', 192, 1, 1, 0), ('conv', 192, (1,7), 1, (0,3)),
             ('conv', 192, (7,1), 1, (3,0)), ('conv', 192, 3, 2, 0)],
            [('max', 3, 2, 0)]
        ]
        return self.inception_module('inception_d', inputs, branches)

    def module_e(self, inputs, pooltype):
        branches = [
            [('conv', 320, 1, 1, 0)],
            [('conv', 384, 1, 1, 0), ('conv', 384, (1,3), 1, (0,1))],
            [('share',),             ('conv', 384, (3,1), 1, (1,0))],
            [('conv', 448, 1, 1, 0), ('conv', 384, 3, 1, 1), ('conv', 384, (1,3), 1, (0,1))],
            [('share',),             ('share',),             ('conv', 384, (3,1), 1, (1,0))],
            [(pooltype, 3, 1, 1), ('conv', 192, 1, 1, 0)]
        ]
        return self.inception_module('inception_e', inputs, branches)

    def __init__(self, params):
        Model.check_parameters(params, {'name': 'InceptionV3'})
        BaseInceptionModel.__init__(self, params)

        v = mx.sym.Variable(name="data")
        # Input conv modules
        v = self.conv('conv', v, num_filters=32, kernel=(3, 3), stride=(2, 2), pad=(0, 0))
        v = self.conv('conv', v, num_filters=32, kernel=(3, 3), stride=(1, 1), pad=(0, 0))
        v = self.conv('conv', v, num_filters=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
        v = mx.symbol.Pooling(name='pool1', data=v, pool_type="max", kernel=(3, 3), stride=(2, 2))
        v = self.conv('conv', v, num_filters=80, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
        v = self.conv('conv', v, num_filters=192, kernel=(3, 3), stride=(1, 1), pad=(0, 0))
        v = mx.symbol.Pooling(name='pool2', data=v, pool_type="max", kernel=(3, 3), stride=(2, 2))
        # Three Type A inception modules
        for sz in (32, 64, 64):
            v = self.module_a(v, sz)
        # One Type B inception module
        v = self.module_b(v)
        # Four Type C inception modules
        for sz in (128, 160, 160, 192):
            v = self.module_c(v, sz)
        # One Type D inception module
        v = self.module_d(v)
        # Two Type E inception modules
        v = self.module_e(v, 'avg')
        v = self.module_e(v, 'max')
        # Final global pooling
        v = mx.symbol.Pooling(name='pool', data=v, pool_type="avg", kernel=(8, 8), stride=(1, 1))
        # And classifier
        self.__output = self.add_head_nodes(v)


class Inception4(BaseInceptionModel):

    implements = 'inception4'

    @property
    def output(self):
        return self.__output

    # Stem functions
    def inception_v4_sa(self, inputs):
        branches = [
            [('conv', 96, 3, 2, 0)],
            [('max', 3, 2, 0)]
        ]
        return self.inception_module('incept_v4_sa', inputs, branches)

    def inception_v4_sb(self, inputs):
        branches = [
            [('conv', 64, 1, 1, 0), ('conv', 96, 3, 1, 0)],
            [('conv', 64, 1, 1, 0), ('conv', 64, (1,7), 1, (0,3)), ('conv', 64, (7,1), 1, (3,0)),
             ('conv', 96, 3, 1, 0)]
        ]
        return self.inception_module('incept_v4_sb', inputs, branches)

    def inception_v4_sc(self, inputs):
        branches = [
            [('conv', 192, 3, 2, 0)],
            [('max', 3, 2, 0)]
        ]
        return self.inception_module('incept_v4_sc', inputs, branches)

    # Reduction functions
    def inception_v4_ra(self, inputs, k, l, m, n):
        branches = [
            [('conv', n, 3, 2, 0)],
            [('conv', k, 1, 1, 0), ('conv', l, 3, 1, 1), ('conv', m, 3, 2, 0)],
            [('max', 3, 2, 0)]
        ]
        return self.inception_module('incept_v4_ra', inputs, branches)

    def inception_v4_rb(self, inputs):
        branches = [
            [('conv', 192, 1, 1, 0), ('conv', 192, 3, 2, 0)],
            [('conv', 256, 1, 1, 0), ('conv', 256, (1,7), 1, (0,3)), ('conv', 320, (7,1), 1, (3,0)),
             ('conv', 320, 3, 2, 0)],
            [('max', 3, 2, 0)],
        ]
        return self.inception_module('incept_v4_rb', inputs, branches)

    def inception_v4_a(self, inputs):
        branches = [
            [('conv', 96, 1, 1, 0)],
            [('conv', 64, 1, 1, 0), ('conv', 96, 3, 1, 1)],
            [('conv', 64, 1, 1, 0), ('conv', 96, 3, 1, 1), ('conv', 96, 3, 1, 1)],
            [('avg', 3, 1, 1), ('conv', 96, 1, 1, 0)]
        ]
        return self.inception_module('incept_v4_a', inputs, branches)

    def inception_v4_b(self, inputs):
        branches = [
            [('conv', 384, 1, 1, 0)],
            [('conv', 192, 1, 1, 0), ('conv', 224, (1,7), 1, (0,3)), ('conv', 256, (7,1), 1, (3,0))],
            [('conv', 192, 1, 1, 0), ('conv', 192, (7,1), 1, (3,0)), ('conv', 224, (1,7), 1, (0,3)),
             ('conv', 224, (7,1), 1, (3,0)), ('conv', 256, (1,7), 1, (0,3))],
            [('avg', 3, 1, 1), ('conv', 128, 1, 1, 0)]
        ]
        return self.inception_module('incept_v4_b', inputs, branches)

    def inception_v4_c(self, inputs):
        branches = [
            [('conv', 256, 1, 1, 0)],
            [('conv', 384, 1, 1, 0), ('conv', 256, (1,3), 1, (0,1))],
            [('share',),             ('conv', 256, (3,1), 1, (1,0))],
            [('conv', 384, 1, 1, 0), ('conv', 448, (3,1), 1, (1,0)),
             ('conv', 512, (1,3), 1, (0,1)), ('conv', 256, (1,3), 1, (0,1))],
            [('share',), ('share',), ('share',), ('conv', 256, (3,1), 1, (1,0))],
            [('avg', 3, 1, 1), ('conv', 256, 1, 1, 0)]
        ]
        return self.inception_module('incept_v4_c', inputs, branches)

    def __init__(self, params):
        Model.check_parameters(params, {'name': 'InceptionV4'})
        BaseInceptionModel.__init__(self, params)

        v = mx.sym.Variable(name="data")
        # Input conv modules
        v = self.conv('conv', v, num_filters=32, kernel=(3, 3), stride=(2, 2), pad=(0, 0))
        v = self.conv('conv', v, num_filters=32, kernel=(3, 3), stride=(1, 1), pad=(0, 0))
        v = self.conv('conv', v, num_filters=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
        # Stem modules
        v = self.inception_v4_sa(v)
        v = self.inception_v4_sb(v)
        v = self.inception_v4_sc(v)
        # Four Type A modules
        for _ in xrange(4):
            v = self.inception_v4_a(v)
        # One Type A Reduction module
        v = self.inception_v4_ra(v, 192, 224, 256, 384)
        # Seven Type B modules
        for _ in xrange(7):
            v = self.inception_v4_b(v)
        # One Type B Reduction module
        v = self.inception_v4_rb(v)
        # Three Type C modules
        for _ in xrange(3):
            v = self.inception_v4_c(v)
        # Final global pooling
        v = mx.symbol.Pooling(name='pool', data=v, pool_type="avg", kernel=(8, 8), stride=(1, 1))
        if self.phase == 'training':
            v = mx.symbol.Dropout(name='drop', data=v, p=0.2)
        # And classifier
        self.__output = self.add_head_nodes(v)
