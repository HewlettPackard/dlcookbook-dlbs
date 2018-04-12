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

# https://github.com/dmlc/mxnet/blob/master/example/image-classification/symbols/googlenet.py
# http://ethereon.github.io/netscope/#/gist/4325909f3683e51eaf93fdaeed6b2a9b

def ConvFactory(data, num_filter, kernel, stride=(1, 1), pad=(0, 0), name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    act = mx.symbol.Activation(data=conv, act_type='relu', name='relu_%s%s' %(name, suffix))
    return act

def InceptionFactory(data, num_1x1, num_3x3red, num_3x3, num_d5x5red, num_d5x5, pool, proj, name):
    # 1x1
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_1x1' % name))
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd5x5r = ConvFactory(data=data, num_filter=num_d5x5red, kernel=(1, 1), name=('%s_5x5' % name), suffix='_reduce')
    cd5x5 = ConvFactory(data=cd5x5r, num_filter=num_d5x5, kernel=(5, 5), pad=(2, 2), name=('%s_5x5' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = ConvFactory(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_proj' %  name))
    # concat
    concat = mx.symbol.Concat(*[c1x1, c3x3, cd5x5, cproj], name='ch_concat_%s_chconcat' % name)
    return concat

class GoogleNet(Model):
    
    implements = 'googlenet'
    
    @property
    def output(self):
        return self.__output

    def __init__(self, params):
        """ Naming and topology according to: http://ethereon.github.io/netscope/#/gist/f2e4825a8d4f8a3609cefd7ffadc910a
            Based on: https://github.com/dmlc/mxnet/blob/master/example/image-classification/symbols/alexnet.py
        """
        Model.check_parameters(
            params,
            {'name': 'GoogleNet', 'input_shape':(3, 224, 224),
             'num_classes': 1000, 'phase': 'training',
             'dtype': 'float32'}
        )
        Model.__init__(self, params)

        if self.dtype == 'float16':
            print("[WARNING] MxNet does not provide half precision kernel for LRN layer. It will be disabled. "\
                  "Thus, comparison with single precision version or other frameworks will not be totally fair.")

        training = self.phase == 'training'
        data = self.add_data_node()

        conv1 = ConvFactory(data, 64, kernel=(7, 7), stride=(2, 2), pad=(3, 3), name="conv1/7x7_s2")
        pool1 = mx.sym.Pooling(conv1, kernel=(3, 3), stride=(2, 2), pool_type="max", name="pool1/3x3_s2")
        norm1 = self.maybe_lrn(pool1, 'pool1/norm1')

        conv2_reduce = ConvFactory(norm1, 64, kernel=(1, 1), stride=(1, 1), name="conv2/3x3_reduce")

        conv2 = ConvFactory(conv2_reduce, 192, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name="conv2/3x3")
        norm2 = self.maybe_lrn(conv2, 'conv2/norm2')
        pool2 = mx.sym.Pooling(norm2, kernel=(3, 3), stride=(2, 2), pool_type="max", name='pool2/3x3_s2')

        in3a = InceptionFactory(pool2, 64, 96, 128, 16, 32, "max", 32, name="inception_3a")
        in3b = InceptionFactory(in3a, 128, 128, 192, 32, 96, "max", 64, name="inception_3b")
        pool3 = mx.sym.Pooling(in3b, kernel=(3, 3), stride=(2, 2), pool_type="max", name='pool3/3x3_s2')

        in4a = InceptionFactory(pool3, 192, 96, 208, 16, 48, "max", 64, name="inception_4a")
        in4b = InceptionFactory(in4a, 160, 112, 224, 24, 64, "max", 64, name="inception_4b")
        in4c = InceptionFactory(in4b, 128, 128, 256, 24, 64, "max", 64, name="inception_4c")
        in4d = InceptionFactory(in4c, 112, 144, 288, 32, 64, "max", 64, name="inception_4d")
        in4e = InceptionFactory(in4d, 256, 160, 320, 32, 128, "max", 128, name="inception_4e")
        pool4 = mx.sym.Pooling(in4e, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type="max", name='pool4/3x3_s2')

        in5a = InceptionFactory(pool4, 256, 160, 320, 32, 128, "max", 128, name="inception_5a")
        in5b = InceptionFactory(in5a, 384, 192, 384, 48, 128, "max", 128, name="inception_5b")
        pool5 = mx.sym.Pooling(in5b, kernel=(7, 7), stride=(1, 1), pool_type="avg", name='pool5/7x7_s1')
        flatten5 = mx.sym.Flatten(data=pool5)
        drop5 = mx.symbol.Dropout(data=flatten5, p=0.5, name='pool5/drop_7x7_s1') if training else flatten5

        self.__output = self.add_head_nodes(drop5)
