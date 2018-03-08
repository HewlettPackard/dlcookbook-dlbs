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
import torch
import torch.nn as nn
from pytorch_benchmarks.models.model import Model


class ConvModule(nn.Module):
    """ [input] -> Conv2D -> BN -> ReLU -> [output] """
    def __init__(self, num_input_channels, num_filters, kernel_size,
                 stride=1, padding=0):
        super(ConvModule, self).__init__()
        self.conv_module = nn.Sequential(
            nn.Conv2d(num_input_channels, num_filters, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(num_filters, eps=2e-5, momentum=0.9, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_module(x)


class ParallelModule(nn.Module):
    """Contains arbitrary number of parallel branches. Each branch is a
       sequential container that may contain parallel modules. All parallel
       modules accept the same input tensor and their outputs are concatenated
       along channel (2nd) dimension indexed as '1'.
    """
    def __init__(self, num_input_channels, branches, module_name):
        """Batch, Channel, Width, Height
        """
        super(ParallelModule, self).__init__()
        if not isinstance(branches, list):
            raise ValueError("Unknown parallel module format - expecting list")
        self.parallel_branches = nn.ModuleList()
        self.num_output_channels = 0
        # Iterate over each parallel branch. Each branch contains one or more
        # operators (module) that can be parallel branches.
        for branch_idx, branch in enumerate(branches):
            branch_name = "%s/b_%d" % (module_name, branch_idx)
            if not isinstance(branch, list):
                msg = "Unknown branch format. Module='%s', branch='%s', branch_type='%s'"
                raise ValueError(msg % (module_name, branch_name, type(branch)))
            # Each parallel branch is a sequential module.
            branch_modules = nn.Sequential()
            # In the end of this loop, num_prev_channels contains number of
            # output channels.
            num_prev_channels = num_input_channels
            # Iterate over operators in this branch
            for op_idx, op in enumerate(branch):
                op_name = "%s/op_%d" % (branch_name, op_idx)
                if isinstance(op, list):
                    op_module = ParallelModule(num_prev_channels, op, op_name)
                    num_prev_channels = op_module.num_output_channels
                elif isinstance(op, tuple):
                    if op[0] == 'conv':
                        op_module = ConvModule(
                            num_prev_channels, num_filters=op[1], kernel_size=op[2],
                            stride=op[3], padding=op[4]
                        )
                        num_prev_channels = op[1]
                    elif op[0] == 'max':
                        op_module = nn.MaxPool2d(
                            kernel_size=op[1], stride=op[2], padding=op[3]
                        )
                    elif op[0] == 'avg':
                        op_module = nn.AvgPool2d(
                            kernel_size=op[1], stride=op[2], padding=op[3]
                        )
                    else:
                        raise ValueError("Unknown operator type (%s)" % op[0])
                else:
                    raise ValueError("Unknown operator format - expecting list or tuple")
                branch_modules.add_module(op_name, op_module)
                # We will stack output tensors along channel dimension
                self.num_output_channels += num_prev_channels
            self.parallel_branches.append(branch_modules)

    def forward(self, x):
        outputs = [None] * len(self.parallel_branches)
        for idx, branch in enumerate(self.parallel_branches):
            outputs[idx] = branch(x)
        return torch.cat(outputs, dim=1)


class BaseInceptionModel(Model):

    def __init__(self, params):
        Model.check_parameters(
            params,
            {'input_shape':(3, 299, 299), 'num_classes': 1000,
             'phase': 'training',
             'dtype': 'float32'}
        )
        Model.__init__(self, params)


class Inception3(BaseInceptionModel):

    implements = 'inception3'

    def module_a(self, num_input_channels, index, n):
        branches = [
            [('conv', 64, 1, 1, 0)],
            [('conv', 48, 1, 1, 0), ('conv', 64, 5, 1, 2)],
            [('conv', 64, 1, 1, 0), ('conv', 96, 3, 1, 1), ('conv', 96, 3, 1, 1)],
            [('avg', 3, 1, 1), ('conv', n, 1, 1, 0)]
        ]
        return ParallelModule(num_input_channels, branches, 'inception_a_%d' % index)

    def module_b(self, num_input_channels, index):
        branches = [
            [('conv', 384, 3, 2, 0)],
            [('conv', 64, 1, 1, 0), ('conv', 96, 3, 1, 1), ('conv', 96, 3, 2, 0)],
            [('max', 3, 2, 0)]
        ]
        return ParallelModule(num_input_channels, branches, 'inception_b_%d' % index)

    def module_c(self, num_input_channels, index, n):
        branches = [
            [('conv', 192, 1, 1, 0)],
            [('conv', n, 1, 1, 0), ('conv', n, (1, 7), 1, (0, 3)), ('conv', 192, (7, 1), 1, (3, 0))],
            [('conv', n, 1, 1, 0), ('conv', n, (7, 1), 1, (3, 0)), ('conv', n, (1, 7), 1, (0, 3)),
             ('conv', n, (7, 1), 1, (3, 0)), ('conv', 192, (1, 7), 1, (0, 3))],
            [('avg', 3, 1, 1), ('conv', 192, 1, 1, 0)]
        ]
        return ParallelModule(num_input_channels, branches, 'inception_c_%d' % index)

    def module_d(self, num_input_channels, index):
        branches = [
            [('conv', 192, 1, 1, 0), ('conv', 320, 3, 2, 0)],
            [('conv', 192, 1, 1, 0), ('conv', 192, (1, 7), 1, (0, 3)),
             ('conv', 192, (7, 1), 1, (3, 0)), ('conv', 192, 3, 2, 0)],
            [('max', 3, 2, 0)]
        ]
        return ParallelModule(num_input_channels, branches, 'inception_d_%d' % index)

    def module_e(self, num_input_channels, index, pooltype):
        branches = [
            [('conv', 320, 1, 1, 0)],
            [('conv', 384, 1, 1, 0), [[('conv', 384, (1, 3), 1, (0, 1))], [('conv', 384, (3, 1), 1, (1, 0))]]],
            [('conv', 448, 1, 1, 0), ('conv', 384, 3, 1, 1), [[('conv', 384, (1, 3), 1, (0, 1))], [('conv', 384, (3, 1), 1, (1, 0))]]],
            [(pooltype, 3, 1, 1), ('conv', 192, 1, 1, 0)]
        ]
        return ParallelModule(num_input_channels, branches, 'inception_e_%d' % index)

    def __init__(self, params):
        Model.check_parameters(params, {'name': 'InceptionV3'})
        BaseInceptionModel.__init__(self, params)
        self.features = nn.Sequential(
            # Input conv modules
            ConvModule(3, num_filters=32, kernel_size=3, stride=2, padding=0),
            ConvModule(32, num_filters=32, kernel_size=3, stride=1, padding=0),
            ConvModule(32, num_filters=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvModule(64, num_filters=80, kernel_size=1, stride=1, padding=0),
            ConvModule(80, num_filters=192, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Three Type A inception modules
            self.module_a(192, index=0, n=32),
            self.module_a(256, index=1, n=64),
            self.module_a(288, index=2, n=64),
            # One Type B inception module
            self.module_b(288, index=0),
            # Four Type C inception modules
            self.module_c(768, index=0, n=128),
            self.module_c(768, index=1, n=160),
            self.module_c(768, index=2, n=160),
            self.module_c(768, index=3, n=192),
            # One Type D inception module
            self.module_d(768, index=0),
            # Two Type E inception modules
            self.module_e(1280, index=0, pooltype='avg'),
            self.module_e(2048, index=1, pooltype='max'),
            # Final global pooling
            nn.AvgPool2d(kernel_size=8, stride=1)
        )
        self.classifier = nn.Sequential(
            # And classifier
            nn.Dropout(p=0.2),
            nn.Linear(2048, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 2048 * 1 * 1)
        return self.classifier(x)


class Inception4(BaseInceptionModel):

    implements = 'inception4'

    # Stem functions
    def inception_v4_sa(self, num_input_channels, index):
        branches = [
            [('conv', 96, 3, 2, 0)],
            [('max', 3, 2, 0)]
        ]
        return ParallelModule(num_input_channels, branches, 'incept_v4_sa_%d' % index)

    def inception_v4_sb(self, num_input_channels, index):
        branches = [
            [('conv', 64, 1, 1, 0), ('conv', 96, 3, 1, 0)],
            [('conv', 64, 1, 1, 0), ('conv', 64, (1, 7), 1, (0, 3)), ('conv', 64, (7, 1), 1, (3, 0)),
             ('conv', 96, 3, 1, 0)]
        ]
        return ParallelModule(num_input_channels, branches, 'incept_v4_sb_%d' % index)

    def inception_v4_sc(self, num_input_channels, index):
        branches = [
            [('conv', 192, 3, 2, 0)],
            [('max', 3, 2, 0)]
        ]
        return ParallelModule(num_input_channels, branches, 'incept_v4_sc_%d' % index)

    # Reduction functions
    def inception_v4_ra(self, num_input_channels, index, k, l, m, n):
        branches = [
            [('conv', n, 3, 2, 0)],
            [('conv', k, 1, 1, 0), ('conv', l, 3, 1, 1), ('conv', m, 3, 2, 0)],
            [('max', 3, 2, 0)]
        ]
        return ParallelModule(num_input_channels, branches, 'incept_v4_ra_%d' % index)

    def inception_v4_rb(self, num_input_channels, index):
        branches = [
            [('conv', 192, 1, 1, 0), ('conv', 192, 3, 2, 0)],
            [('conv', 256, 1, 1, 0), ('conv', 256, (1, 7), 1, (0, 3)), ('conv', 320, (7, 1), 1, (3, 0)),
             ('conv', 320, 3, 2, 0)],
            [('max', 3, 2, 0)],
        ]
        return ParallelModule(num_input_channels, branches, 'incept_v4_rb_%d' % index)

    def inception_v4_a(self, num_input_channels, index):
        branches = [
            [('conv', 96, 1, 1, 0)],
            [('conv', 64, 1, 1, 0), ('conv', 96, 3, 1, 1)],
            [('conv', 64, 1, 1, 0), ('conv', 96, 3, 1, 1), ('conv', 96, 3, 1, 1)],
            [('avg', 3, 1, 1), ('conv', 96, 1, 1, 0)]
        ]
        return ParallelModule(num_input_channels, branches, 'incept_v4_a_%d' % index)

    def inception_v4_b(self, num_input_channels, index):
        branches = [
            [('conv', 384, 1, 1, 0)],
            [('conv', 192, 1, 1, 0), ('conv', 224, (1, 7), 1, (0, 3)), ('conv', 256, (7, 1), 1, (3, 0))],
            [('conv', 192, 1, 1, 0), ('conv', 192, (7, 1), 1, (3, 0)), ('conv', 224, (1, 7), 1, (0, 3)),
             ('conv', 224, (7, 1), 1, (3, 0)), ('conv', 256, (1, 7), 1, (0, 3))],
            [('avg', 3, 1, 1), ('conv', 128, 1, 1, 0)]
        ]
        return ParallelModule(num_input_channels, branches, 'incept_v4_b_%d' % index)

    def inception_v4_c(self, num_input_channels, index):
        branches = [
            [('conv', 256, 1, 1, 0)],
            [('conv', 384, 1, 1, 0), [[('conv', 256, (1, 3), 1, (0, 1))], [('conv', 256, (3, 1), 1, (1, 0))]]],
            [('conv', 384, 1, 1, 0), ('conv', 448, (3, 1), 1, (1, 0)), ('conv', 512, (1, 3), 1, (0, 1)),
             [[('conv', 256, (1, 3), 1, (0, 1))],[('conv', 256, (3, 1), 1, (1, 0))]]],
            [('avg', 3, 1, 1), ('conv', 256, 1, 1, 0)]
        ]
        return ParallelModule(num_input_channels, branches, 'incept_v4_b_%d' % index)

    def __init__(self, params):
        Model.check_parameters(params, {'name': 'InceptionV4'})
        BaseInceptionModel.__init__(self, params)

        self.features = nn.Sequential(
            # Input conv modules
            ConvModule(3, num_filters=32, kernel_size=3, stride=2, padding=0),
            ConvModule(32, num_filters=32, kernel_size=3, stride=1, padding=0),
            ConvModule(32, num_filters=64, kernel_size=3, stride=1, padding=1),
            # Stem modules
            self.inception_v4_sa(64, index=0),
            self.inception_v4_sb(160, index=0),
            self.inception_v4_sc(192, index=0),
            # Four Type A modules
            self.inception_v4_a(384, index=0),
            self.inception_v4_a(384, index=1),
            self.inception_v4_a(384, index=2),
            self.inception_v4_a(384, index=3),
            # One Type A Reduction module
            self.inception_v4_ra(384, 0, 192, 224, 256, 384),
            # Seven Type B modules
            self.inception_v4_b(1024, index=0),
            self.inception_v4_b(1024, index=1),
            self.inception_v4_b(1024, index=2),
            self.inception_v4_b(1024, index=3),
            self.inception_v4_b(1024, index=4),
            self.inception_v4_b(1024, index=5),
            self.inception_v4_b(1024, index=6),
            # One Type B Reduction module
            self.inception_v4_rb(1024, index=0),
            # Three Type C modules
            self.inception_v4_c(1536, index=0),
            self.inception_v4_c(1536, index=1),
            self.inception_v4_c(1536, index=2),
            # Final global pooling
            nn.AvgPool2d(kernel_size=8, stride=1)
        )
        self.classifier = nn.Sequential(
            # And classifier
            nn.Dropout(p=0.2),
            nn.Linear(1536, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 1536 * 1 * 1)
        return self.classifier(x)
