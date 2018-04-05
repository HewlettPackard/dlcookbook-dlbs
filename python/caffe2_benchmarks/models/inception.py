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
from caffe2.python import brew
from collections import defaultdict
from caffe2_benchmarks.models.model import Model


class BaseInceptionModel(Model):
    """Inception neural network model."""

    def conv(self, model, name, inputs, input_depth, num_filters, kernel, stride,
             pad, is_inference):
        # Check padding
        if isinstance(pad, int):
            pad_t = pad_b = pad_l = pad_r = pad
        elif isinstance(pad, list) or isinstance(pad, tuple):
            if len(pad) == 2:
                pad_t = pad_b = pad[0]
                pad_l = pad_r = pad[1]
            elif len(pad) == 4:
                pad_t = pad[0]
                pad_b = pad[1]
                pad_l = pad[2]
                pad_r = pad[3]
            else:
                assert False, "Invalid length of pad array. Expecting 2 or 4 but have: " + str(pad)
        else:
            assert False, "Invalid type of padding: " + str(pad)
        # Check kernel
        if isinstance(kernel, int):
            kernel = [kernel, kernel]
        elif isinstance(kernel, tuple) or isinstance(kernel, list):
            assert len(kernel) == 2, "Kernel must have length 2"
            kernel = [kernel[0], kernel[1]]
        else:
            assert False, "Invalid type of kerne;: " + str(kernel)
        #
        self.counts[name] += 1
        name = name + str(self.counts[name]-1)
        #
        v = brew.conv(model, inputs, name + '_conv', input_depth, num_filters,
                      kernel=kernel, stride=stride,
                      pad_t=pad_t, pad_l=pad_l, pad_b=pad_b, pad_r=pad_r,
                      no_bias=True)
        v = brew.spatial_bn(model, v, name+'_bn', num_filters, eps=2e-5,
                            momentum=0.9, is_test=is_inference)
        v = brew.relu(model, v, name+'_relu')
        return v

    def inception_module(self, model, name, inputs, input_depth, branches, is_inference):
        """Add parallel branches from 'branhes' into current graph"""
        self.counts[name] += 1
        name = name + str(self.counts[name]-1)
        layers_outputs = []             # Outputs of each layer in each branch
        layers_input_depths = []        # Layers input depths
        for branch_id, branch in enumerate(branches):
            v = inputs                       # Inputs for a next layer in a branch
            last_layer_output_depth = input_depth
            layers_outputs.append([])        # Outputs of layers in this branch
            layers_input_depths.append([0]*len(branch))
            for layer_id, layer in enumerate(branch):
                layers_input_depths[-1][layer_id] = last_layer_output_depth
                if layer[0] == 'conv':
                    v = self.conv(
                        model,
                        name='%s_b%d_l%d_' % (name, branch_id, layer_id),
                        inputs=v,
                        input_depth=last_layer_output_depth,
                        num_filters=layer[1],
                        kernel=layer[2],
                        stride=layer[3],
                        pad=layer[4],
                        is_inference=is_inference
                    )
                    last_layer_output_depth = layer[1]
                elif layer[0] == 'avg' or layer[0] == 'max':
                    pool_func = brew.average_pool if layer[0] == 'avg' else brew.max_pool
                    v = pool_func(
                        model, v,
                        blob_out='%s_b%d_p%d' % (name, branch_id, layer_id),
                        kernel=layer[1], stride=layer[2], pad=layer[3]
                    )
                    # last_layer_output_depth does not change here
                elif layer[0] == 'share':
                    v = layers_outputs[-2][layer_id]
                    last_layer_output_depth = layers_input_depths[-2][layer_id+1]
                else:
                    assert False, 'Unknown later type - ' + layer[0]
                layers_outputs[-1].append(v)
        # concat
        concat = brew.concat(
            model,
            [outputs[-1] for outputs in layers_outputs],
            blob_out='%s_concat' % name
        )
        return concat

    def __init__(self, params):
        Model.check_parameters(
            params,
            {'input_shape':((3, 299, 299)),
             'num_classes': 1000, 'arg_scope': {'order': 'NCHW'}}
        )
        Model.__init__(self, params)
        self.counts = defaultdict(lambda: 0)


class Inception3(BaseInceptionModel):
    """Inception neural network model."""

    implements = 'inception3'

    def module_a(self, model, inputs, input_depth, n, is_inference):
        branhes = [
            [('conv', 64, 1, 1, 0)],
            [('conv', 48, 1, 1, 0), ('conv', 64, 5, 1, 2)],
            [('conv', 64, 1, 1, 0), ('conv', 96, 3, 1, 1), ('conv', 96, 3, 1, 1)],
            [('avg', 3, 1, 1), ('conv', n, 1, 1, 0)]
        ]
        return self.inception_module(model, 'inception_a', inputs, input_depth,
                                     branhes, is_inference)

    def module_b(self, model, inputs, input_depth, is_inference):
        branches = [
            [('conv', 384, 3, 2, 0)],
            [('conv', 64, 1, 1, 0), ('conv', 96, 3, 1, 1), ('conv', 96, 3, 2, 0)],
            [('max', 3, 2, 0)]
        ]
        return self.inception_module(model, 'inception_b', inputs, input_depth,
                                     branches, is_inference)

    def module_c(self, model, inputs, input_depth, n, is_inference):
        branches = [
            [('conv', 192, 1, 1, 0)],
            [('conv', n, 1, 1, 0), ('conv', n, (1,7), 1, (0,3)), ('conv', 192, (7,1), 1, (3,0))],
            [('conv', n, 1, 1, 0), ('conv', n, (7,1), 1, (3,0)), ('conv', n, (1,7), 1, (0,3)),
             ('conv', n, (7,1), 1, (3,0)), ('conv', 192, (1,7), 1, (0,3))],
            [('avg', 3, 1, 1), ('conv', 192, 1, 1, 0)]
        ]
        return self.inception_module(model, 'inception_c', inputs, input_depth,
                                     branches, is_inference)

    def module_d(self, model, inputs, input_depth, is_inference):
        branches = [
            [('conv', 192, 1, 1, 0), ('conv', 320, 3, 2, 0)],
            [('conv', 192, 1, 1, 0), ('conv', 192, (1,7), 1, (0,3)),
             ('conv', 192, (7,1), 1, (3,0)), ('conv', 192, 3, 2, 0)],
            [('max', 3, 2, 0)]
        ]
        return self.inception_module(model, 'inception_d', inputs, input_depth,
                                     branches, is_inference)

    def module_e(self, model, inputs, input_depth, pooltype, is_inference):
        branches = [
            [('conv', 320, 1, 1, 0)],
            [('conv', 384, 1, 1, 0), ('conv', 384, (1,3), 1, (0,1))],
            [('share',),             ('conv', 384, (3,1), 1, (1,0))],
            [('conv', 448, 1, 1, 0), ('conv', 384, 3, 1, 1), ('conv', 384, (1,3), 1, (0,1))],
            [('share',),             ('share',),             ('conv', 384, (3,1), 1, (1,0))],
            [(pooltype, 3, 1, 1), ('conv', 192, 1, 1, 0)]
        ]
        return self.inception_module(model, 'inception_e', inputs, input_depth,
                                     branches, is_inference)

    def __init__(self, params):
        Model.check_parameters(params, {'name': 'InceptionV3'})
        BaseInceptionModel.__init__(self, params)

    def forward_pass_builder(self, model, loss_scale=1.0):
        """
            This function adds the operators, layers to the network. It should return a list
            of loss-blobs that are used for computing the loss gradient. This function is
            also passed an internally calculated loss_scale parameter that is used to scale
            your loss to normalize for the number of GPUs. Signature: function(model, loss_scale)
        """
        self.counts = defaultdict(lambda: 0)
        is_inference = self.phase == 'inference'

        v = 'data'

        # Input conv modules
        v = self.conv(model, 'conv', v, input_depth=3, num_filters=32, kernel=3, stride=2, pad=0, is_inference=is_inference)
        v = self.conv(model, 'conv', v, input_depth=32, num_filters=32, kernel=3, stride=1, pad=0, is_inference=is_inference)
        v = self.conv(model, 'conv', v, input_depth=32, num_filters=64, kernel=3, stride=1, pad=1, is_inference=is_inference)
        v = brew.max_pool(model, v, blob_out='pool1', kernel=3, stride=2, pad=0)
        v = self.conv(model, 'conv', v, input_depth=64, num_filters=80, kernel=1, stride=1, pad=0, is_inference=is_inference)
        v = self.conv(model, 'conv', v, input_depth=80, num_filters=192, kernel=3, stride=1, pad=0, is_inference=is_inference)
        v = brew.max_pool(model, v, blob_out='pool2', kernel=3, stride=2, pad=0)
        # Three Type A inception modules
        v = self.module_a(model, inputs=v, input_depth=192, n=32, is_inference=is_inference)
        v = self.module_a(model, inputs=v, input_depth=256, n=64, is_inference=is_inference)
        v = self.module_a(model, inputs=v, input_depth=288, n=64, is_inference=is_inference)
        # One Type B inception module
        v = self.module_b(model, inputs=v, input_depth=288, is_inference=is_inference)
        # Four Type C inception modules
        for n in (128, 160, 160, 192):
            v = self.module_c(model, inputs=v, input_depth=768, n=n, is_inference=is_inference)
        # One Type D inception module
        v = self.module_d(model, inputs=v, input_depth=768, is_inference=is_inference)
        # Two Type E inception modules
        v = self.module_e(model, inputs=v, input_depth=1280, pooltype='avg', is_inference=is_inference)
        v = self.module_e(model, inputs=v, input_depth=2048, pooltype='max', is_inference=is_inference)
        # Final global pooling
        v = brew.average_pool(model, v, blob_out='pool', kernel=8, stride=1, pad=0)
        # And classifier
        return self.add_head_nodes(model, v, 2048, 'classifier', loss_scale=loss_scale)


class Inception4(BaseInceptionModel):
    """Inception neural network model."""

    implements = 'inception4'

    # Stem functions
    def inception_v4_sa(self, model, inputs, input_depth, is_inference):
        branches = [
            [('conv', 96, 3, 2, 0)],
            [('max', 3, 2, 0)]
        ]
        return self.inception_module(model, 'incept_v4_sa', inputs, input_depth,
                                     branches, is_inference)

    def inception_v4_sb(self, model, inputs, input_depth, is_inference):
        branches = [
            [('conv', 64, 1, 1, 0), ('conv', 96, 3, 1, 0)],
            [('conv', 64, 1, 1, 0), ('conv', 64, (1,7), 1, (0,3)), ('conv', 64, (7,1), 1, (3,0)),
             ('conv', 96, 3, 1, 0)]
        ]
        return self.inception_module(model, 'incept_v4_sb', inputs, input_depth,
                                     branches, is_inference)

    def inception_v4_sc(self, model, inputs, input_depth, is_inference):
        branches = [
            [('conv', 192, 3, 2, 0)],
            [('max', 3, 2, 0)]
        ]
        return self.inception_module(model, 'incept_v4_sc', inputs, input_depth,
                                     branches, is_inference)

    # Reduction functions
    def inception_v4_ra(self, model, inputs, input_depth, k, l, m, n, is_inference):
        branches = [
            [('conv', n, 3, 2, 0)],
            [('conv', k, 1, 1, 0), ('conv', l, 3, 1, 1), ('conv', m, 3, 2, 0)],
            [('max', 3, 2, 0)]
        ]
        return self.inception_module(model, 'incept_v4_ra', inputs, input_depth,
                                     branches, is_inference)

    def inception_v4_rb(self, model, inputs, input_depth, is_inference):
        branches = [
            [('conv', 192, 1, 1, 0), ('conv', 192, 3, 2, 0)],
            [('conv', 256, 1, 1, 0), ('conv', 256, (1,7), 1, (0,3)), ('conv', 320, (7,1), 1, (3,0)),
             ('conv', 320, 3, 2, 0)],
            [('max', 3, 2, 0)],
        ]
        return self.inception_module(model, 'incept_v4_rb', inputs, input_depth,
                                     branches, is_inference)

    def inception_v4_a(self, model, inputs, input_depth, is_inference):
        branches = [
            [('conv', 96, 1, 1, 0)],
            [('conv', 64, 1, 1, 0), ('conv', 96, 3, 1, 1)],
            [('conv', 64, 1, 1, 0), ('conv', 96, 3, 1, 1), ('conv', 96, 3, 1, 1)],
            [('avg', 3, 1, 1), ('conv', 96, 1, 1, 0)]
        ]
        return self.inception_module(model, 'incept_v4_a', inputs, input_depth,
                                     branches, is_inference)

    def inception_v4_b(self, model, inputs, input_depth, is_inference):
        branches = [
            [('conv', 384, 1, 1, 0)],
            [('conv', 192, 1, 1, 0), ('conv', 224, (1,7), 1, (0,3)), ('conv', 256, (7,1), 1, (3,0))],
            [('conv', 192, 1, 1, 0), ('conv', 192, (7,1), 1, (3,0)), ('conv', 224, (1,7), 1, (0,3)),
             ('conv', 224, (7,1), 1, (3,0)), ('conv', 256, (1,7), 1, (0,3))],
            [('avg', 3, 1, 1), ('conv', 128, 1, 1, 0)]
        ]
        return self.inception_module(model, 'incept_v4_b', inputs, input_depth,
                                     branches, is_inference)

    def inception_v4_c(self, model, inputs, input_depth, is_inference):
        branches = [
            [('conv', 256, 1, 1, 0)],
            [('conv', 384, 1, 1, 0), ('conv', 256, (1,3), 1, (0,1))],
            [('share',),             ('conv', 256, (3,1), 1, (1,0))],
            [('conv', 384, 1, 1, 0), ('conv', 448, (3,1), 1, (1,0)),
             ('conv', 512, (1,3), 1, (0,1)), ('conv', 256, (1,3), 1, (0,1))],
            [('share',), ('share',), ('share',), ('conv', 256, (3,1), 1, (1,0))],
            [('avg', 3, 1, 1), ('conv', 256, 1, 1, 0)]
        ]
        return self.inception_module(model, 'incept_v4_c', inputs, input_depth,
                                     branches, is_inference)

    def __init__(self, params):
        Model.check_parameters(params, {'name': 'InceptionV4'})
        BaseInceptionModel.__init__(self, params)

    def forward_pass_builder(self, model, loss_scale=1.0):
        """
            This function adds the operators, layers to the network. It should return a list
            of loss-blobs that are used for computing the loss gradient. This function is
            also passed an internally calculated loss_scale parameter that is used to scale
            your loss to normalize for the number of GPUs. Signature: function(model, loss_scale)
        """
        self.counts = defaultdict(lambda: 0)
        is_inference = self.phase == 'inference'

        v = 'data'

        # Input conv modules
        v = self.conv(model, 'conv', v, input_depth=3, num_filters=32, kernel=3, stride=2, pad=0, is_inference=is_inference)
        v = self.conv(model, 'conv', v, input_depth=32, num_filters=32, kernel=3, stride=1, pad=0, is_inference=is_inference)
        v = self.conv(model, 'conv', v, input_depth=32, num_filters=64, kernel=3, stride=1, pad=1, is_inference=is_inference)
        # Stem modules
        v = self.inception_v4_sa(model, inputs=v, input_depth=64, is_inference=is_inference)
        v = self.inception_v4_sb(model, inputs=v, input_depth=160, is_inference=is_inference)
        v = self.inception_v4_sc(model, inputs=v, input_depth=192, is_inference=is_inference)
        # Four Type A modules
        for _ in xrange(4):
            v = self.inception_v4_a(model, inputs=v, input_depth=384, is_inference=is_inference)
        # One Type A Reduction module
        v = self.inception_v4_ra(model, inputs=v, input_depth=384, k=192, l=224, m=256, n=384, is_inference=is_inference)
        # Seven Type B modules
        for _ in xrange(7):
            v = self.inception_v4_b(model, inputs=v, input_depth=1024, is_inference=is_inference)
        # One Type B Reduction module
        v = self.inception_v4_rb(model, inputs=v, input_depth=1024, is_inference=is_inference)
        # Three Type C modules
        for _ in xrange(3):
            v = self.inception_v4_c(model, inputs=v, input_depth=1536, is_inference=is_inference)
        # Final global pooling
        v = brew.average_pool(model, v, blob_out='pool', kernel=8, stride=1, pad=0)
        v = brew.dropout(model, v, 'drop', ratio=0.2, is_test=is_inference)
        # And classifier
        return self.add_head_nodes(model, v, 1536, 'classifier', loss_scale=loss_scale)
