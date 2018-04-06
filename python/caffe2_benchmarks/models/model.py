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
"""Base class for all Caffe2 models."""
from caffe2.proto import caffe2_pb2
from caffe2.python import core
from caffe2.python import workspace
from caffe2.python import brew
from caffe2.python import memonger

class Model(object):
    """Base class for all models."""

    def __init__(self, params):
        """ Sets model parameters.

        :param dict params: Dictionary with parameters. Must contain `name`, `batch_size`,
                            `input_shape`, `num_classes` and `phase`.
        """
        for param in ['name', 'batch_size', 'input_shape', 'num_classes', 'phase', 'dtype']:
            assert param in params, "Missing mandatory neural net parameter '%s'" % param
        assert params['phase'] in ['inference', 'training'],\
               "Invalid phase: '%s'. Expecting 'inference' or 'training'" % (params['phase'])
        if params['dtype'] not in ('float', 'float32', 'float16'):
            print("[WARNING] Suspicious data type '%s'. Expecting 'float', 'float32' or 'float16'." % params['dtype'])

        self.__name = params['name']
        self.__batch_size = params['batch_size']
        self.__input_shape = params['input_shape']
        self.__num_classes = params['num_classes']
        self.__phase = params['phase']
        self.__dtype = params['dtype']

    @staticmethod
    def check_parameters(params, default_params):
        """Ensures `params` dictionary contains all keys in `default_params`

        :param dict params: Dictionary to check.
        :param dict default_params: Values with these keys must present in `params`.
        """
        for param, value in default_params.items():
            if param not in params:
                params[param] = value

    @staticmethod
    def get_device_option(gpu=None):
        """Constructs `core.DeviceOption` object

        :param int gpu: Identifier of GPU to use or None for CPU.
        :return: Instance of `core.DeviceOption`.
        """
        dev_opt = None
        if gpu is None:
            dev_opt = core.DeviceOption(caffe2_pb2.CPU)
        else:
            assert workspace.has_gpu_support, "Workspace does not support GPUs"
            assert gpu >= 0 and gpu < workspace.NumCudaDevices(),\
                   "Workspace does not provide this gpu (%d). "\
                   "Number of GPUs is %d" % (gpu, workspace.NumCudaDevices())
            dev_opt = core.DeviceOption(caffe2_pb2.CUDA, gpu)
        return dev_opt

    @staticmethod
    def add_parameter_update_ops(model):
        """A simple parameter update code.

        :param model_helper.ModelHelper model: Model to add update parameters operators for.
        """
        iteration = brew.iter(model, "ITER")
        learning_rate = model.net.LearningRate([iteration], "LR", base_lr=0.01, policy="fixed")
        one = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
        for param in model.GetParams():
            grad = model.param_to_grad[param]
            model.WeightedSum([param, one, grad, learning_rate], param)

    @staticmethod
    def optimize_gradient_memory(model, loss):
        """A naive implementation of memory optimization

        :param model_helper.ModelHelper model: Model to add update parameters operators for.
        :param list loss: A list of losses.
        """

        model.net._net = memonger.share_grad_blobs(
            model.net,
            loss,
            set(model.param_to_grad.values()),
            namescope='',
            share_activations=False,
        )


    def add_synthetic_inputs(self, model, add_labels=True):
        """Build data ingestion operators for this model for multi-GPU training

        :param model_helper.ModelHelper model: Model to add update parameters operators for.
        """
        # https://caffe2.ai/doxygen-python/html/core_8py_source.html#l00222
        # https://github.com/caffe2/caffe2/blob/master/caffe2/python/examples/resnet50_trainer.py
        suffix = '_fp32' if self.dtype == 'float16' else ''
        # Input data tensor with name 'data' if it's float32 regime or 'data_fp32' if it's float16.
        # If it's float16, we will then convert 'data_fp32' to 'data' tensor with precision float16.
        model.param_init_net.GaussianFill(
            [],
            ['data' + suffix],
            shape=(self.batch_size,) + self.input_shape
        )
        if self.dtype == 'float16':
            print("[INFO] Using 'float16' data type (converting input tensor with synthetic data to float16 tensor).")
            model.param_init_net.FloatToHalf('data' + suffix, 'data')

        if add_labels is True:
            model.param_init_net.ConstantFill(
                [],
                ["softmax_label"],
                shape=[self.batch_size],
                value=1,
                dtype=core.DataType.INT32,
            )
        # data = model.StopGradient(data, data)

    def add_data_inputs(self, model, reader, use_gpu_transform, num_decode_threads = 1):
        """Adds real data pipeline.

        Multiple GPUs in one box will share the same reader masking sure they use
        different data.

        :param model_helper.ModelHelper model: Model to add update parameters operators for.
        :param obj reader: Something returned by model.CreateDB(...)
        :param int num_decode_threads: Number of image decoding threads. For deep computationally
                                       expensive models this can be as small as 1. For high
                                       throughput models such as AlexNetOWT a value of 6-8 for 4
                                       GPUs seems to be reasonable (Voltas, ~9k images/second)
        """
        data, _ = brew.image_input(       # data, label
            model,
            [reader],
            ["data", "softmax_label"],
            batch_size=self.batch_size,   # Per device batch size
            output_type=self.dtype,       # "float" or "float16"
            use_gpu_transform=use_gpu_transform,
            use_caffe_datum=True,
            mean=128.,
            #std=128.,
            scale=self.input_shape[2],
            crop=self.input_shape[2],
            mirror=True,
            is_test=False,
            decode_threads=num_decode_threads
        )
        data = model.StopGradient(data, data)

    def add_head_nodes(self, model, v, dim_in, fc_name, loss_scale=1.0):
        """Adds dense and softmax head nodes.

        :param model_helper.ModelHelper model: Current model to use.
        :param obj v: Input blobs.
        :param int dim_in: Number of input features.
        :param str fc_name: Name of a fully connected operator.
        :param float loss_scale: For multi-GPU case.
        :return: List with one head node. A softmax node if `phase` is `inference`
                 else `loss`.
        """
        v = brew.fc(model, v, fc_name, dim_in=dim_in, dim_out=self.num_classes)
        if self.dtype == 'float16':
            print("[INFO] Converting logits from float16 to float32 for softmax layer")
            v = model.net.HalfToFloat(v, v + '_fp32')
        if self.phase == 'inference':
            softmax = brew.softmax(model, v, 'softmax')
            head_nodes = [softmax]
        else:
            softmax, loss = model.SoftmaxWithLoss([v, 'softmax_label'], ['softmax', 'loss'])
            prefix = model.net.Proto().name
            loss = model.Scale(loss, prefix + "_loss", scale=loss_scale)
            head_nodes = [loss]
        return head_nodes

    @property
    def name(self):
        """Get name of a model"""
        return self.__name

    @property
    def batch_size(self):
        """Get current batch size"""
        return self.__batch_size

    @property
    def input_shape(self):
        """Get shape of input tensor excluding batch size dimension"""
        return self.__input_shape if isinstance(self.__input_shape, tuple)\
               else (self.__input_shape,)

    @property
    def num_classes(self):
        """Get number of classes"""
        return self.__num_classes

    @property
    def phase(self):
        """Get phase ('training' or 'inference')"""
        return self.__phase

    @property
    def dtype(self):
        """Get data type (precision) ('float' or 'float16')"""
        return self.__dtype
