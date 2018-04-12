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
"""Base class for all models"""
import mxnet as mx
import numpy as np

class Model(object):
    """Base class for all models"""

    def __init__(self, params):
        """ name: printable name like AlexNet, ResNet152 etc
            input_shape: tuple of dimensions of input data excluding batch, for
                         instance, (3,224,224) - 3 channels with 224 spatial dimensions
            num_classes: size of output softmax (affine) operator
            phase: 'inference' or 'training'
        """
        for param in ['name', 'input_shape', 'num_classes', 'phase', 'dtype']:
            assert param in params, "Missing mandatory neural net parameter '%s'" % param
        assert params['phase'] in ['inference', 'training'],\
               "Invalid phase: '%s'. Expecting 'inference' or 'training'" % (params['phase'])
        self.__name = params['name']
        self.__input_shape = params['input_shape']
        self.__num_classes = params['num_classes']
        self.__phase = params['phase']
        self.__dtype = params['dtype']

    @staticmethod
    def check_parameters(params, default_params):
        """Ensures `params` dictionary contains all keys in `default_params`

        Args:
            params (dict): Dictionary to check.
            default_params (dict): Values with these keys must present in `params`.
        """
        for param, value in default_params.items():
            if param not in params:
                params[param] = value

    def add_data_node(self, name='data'):
        """Add data node casting it to float16 is required"""
        data = mx.sym.Variable(name=name)
        if self.dtype == 'float16':
            print("Casting input DATA tensor to np.float16")
            data = mx.sym.cast(data=data, dtype=np.float16)
        return data

    def add_head_nodes(self, v):
        """Adds dense and softmax head nodes.

        Args:
            v (obj): input tensor.

        Returns:
            Output tensor
        """
        v = mx.sym.FullyConnected(data=v, num_hidden=self.num_classes)
        if self.dtype == 'float16':
            print("Casting logits to np.float32")
            v = mx.sym.cast(data=v, dtype=np.float32)
        if self.phase == 'training':
            labels = mx.sym.Variable(name="softmax_label")
            v = mx.symbol.SoftmaxOutput(data=v, label=labels, name='softmax')
        else:
            v = mx.symbol.softmax(data=v, name='softmax')
        return v

    def maybe_lrn(v, name):
        """ MxNet does not have float16 kernel for LRN operator. So, we use it only
            for float32 data type. That makes comparison not fair. Need to do something
            about it like dropping completely these operators.
            They are used by AlexNet and GoogleNet.

            :param obj v: Input tensor.
            :param str name: Name of the LRN operator.
            :return: The input tensor 'v' if data type is float16 else result of LRN
                     operator
        """
        if self.dtype == 'float32':
            return mx.symbol.LRN(data=v, alpha=0.0001, beta=0.75, knorm=2, nsize=5, name=name)
        else:
            return v

    @property
    def name(self):
        """Get model name"""
        return self.__name

    @property
    def input_shape(self):
        """Get input shape excluding batch size dimension"""
        return self.__input_shape if isinstance(self.__input_shape, tuple)\
               else (self.__input_shape,)

    @property
    def num_classes(self):
        """Get number of classes"""
        return self.__num_classes

    @property
    def phase(self):
        """Get current phase ('training' or 'inference')"""
        return self.__phase

    @property
    def dtype(self):
        """Get type of data ('float32' or 'float16' or 'int8')"""
        return self.__dtype
