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
from __future__ import print_function
import os
import copy
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
        for param in ['name', 'input_shape', 'num_classes', 'phase', 'dtype', 'model_opts']:
            assert param in params, "Missing mandatory neural net parameter '%s'" % param
        assert params['phase'] in ['inference', 'training'],\
               "Invalid phase: '%s'. Expecting 'inference' or 'training'" % (params['phase'])
        self.__name = params['name']
        self.__input_shape = params['input_shape']
        self.__num_classes = params['num_classes']
        self.__phase = params['phase']
        self.__dtype = params['dtype']
        self.__model_opts = copy.deepcopy(params['model_opts'])
        self.__have_float16_lrn = 'DLBS_MXNET_NO_FLOAT16_LRN' not in os.environ
        self._eval_metric = 'acc'
        # The following two parameters are used by data providers.
        self._labels_shape = (1,)                      # Shape of labels tensor excluding leading batch dimension
        self._labels_range = (0, self.num_classes-1)   # Possible labels' values inclusive
        if self.__dtype == 'float16' and self.__have_float16_lrn:
            print("[WARNING] The data type is 'float16' and I assume MXNET provides a float16 kernel for LRN layer. " \
                  "If this model uses LRN and your MXNET version is outdated, you will get error. In this case, to "\
                  "disable LRN layers in float16 regime, define the following variable 'DLBS_MXNET_NO_FLOAT16_LRN' " \
                  "(the value of this variable does not matter) i.e.: "\
                  "-Pruntime.launcher='\"DLBS_MXNET_NO_FLOAT16_LRN=1 \"'")
        if self.__dtype == 'float16' and not self.__have_float16_lrn:
            print("[WARNING] The data type is 'float16' and you disable LRN layers. All calls to Model.maybe_lrn " \
                  " will do nothing. If your MXNET version is up to date and provides LRN float16 kernel make sure "\
                  "DLBS_MXNET_NO_FLOAT16_LRN environment variable is not defined. All this is relevant only if this "\
                  "model uses LRN operators.")

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
            # Just in case labels are of shape (batch_size, 1) we need to
            # reshape them to (batch_size,).
            labels = mx.sym.Reshape(labels, shape=(-1,))
            v = mx.symbol.SoftmaxOutput(data=v, label=labels, name='softmax')
        else:
            v = mx.symbol.softmax(data=v, name='softmax')
        return v

    def maybe_lrn(self, v, name):
        """ MxNet does not have float16 kernel for LRN operator. So, we use it only
            for float32 data type. That makes comparison not fair. Need to do something
            about it like dropping completely these operators.
            They are used by AlexNet and GoogleNet.

            UPDATE: Seems like mxnet now provides this kernel.

            :param obj v: Input tensor.
            :param str name: Name of the LRN operator.
            :return: The input tensor 'v' if data type is float16 else result of LRN
                     operator
        """
        if self.dtype == 'float32' or self.__have_float16_lrn:
            return mx.symbol.LRN(data=v, alpha=0.0001, beta=0.75, knorm=2, nsize=5, name=name)
        else:
            return v

    def render_to_file(self, node, bsize, fname):
        """Render the neural network to JPG file.

        :param sym node: Head node.
        :param int bsize: Batch size.
        :param str fname: File name without extension.
        """
        g = mx.viz.plot_network(
            node,
            shape={'data': (bsize,) + self.input_shape},
            node_attrs={"shape":'rect', "fixedsize":'false'},
            save_format = 'jpg'
        )
        g.render(fname)

    @staticmethod
    def num_parameters(module, count_aux_params=True):
        """Return number of parameters in a module.
        """
        arg_params, aux_params = module.get_params()
        num_params = 0
        for p in arg_params:
            num_params += np.prod(arg_params[p].shape)
        if count_aux_params:
            for p in aux_params:
                num_params += np.prod(aux_params[p].shape)
        return num_params

    @staticmethod
    def print_parameters(module):
        def __print(params):
            total_params = 0
            pnames = params.keys()
            pnames.sort()
            for p in pnames:
                nparams = np.prod(params[p].shape)
                total_params += nparams
                print("%-30s %-30s %d" % (p, str(params[p].shape), nparams))
            return total_params
                
        arg_params, aux_params = module.get_params()
        print("Arg parameters")
        net_params =  __print(arg_params)
        print("Aux parameters")
        net_params += __print(aux_params)
        print("Total number of parameters %d" % (net_params))

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

    @property
    def model_opts(self):
        """Get additional model options (json dictionary)"""
        return self.__model_opts

    @property
    def eval_metric(self):
        """Return evaluation metric"""
        return self._eval_metric

    @property
    def labels_shape(self):
        """Shape of labels tensor excluding leading batch dimension"""
        return self._labels_shape

    @property
    def labels_range(self):
        """Get range for possible label values. Range is inclusive."""
        return self._labels_range
