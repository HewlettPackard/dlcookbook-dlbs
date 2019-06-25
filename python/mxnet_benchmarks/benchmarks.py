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
"""MXNet benchmarks entry point module.
    A default implementation for mxnet benchmarking environment.
    Based on this Image Classification example:
        https://github.com/apache/incubator-mxnet/blob/master/example/image-classification
    TODO: Think about using that code to do benchmarking?

This module can run from command line or from python application. To benchmark a particular
model on a synthetic data, run the following commands:

.. code-block:: python

    import numpy as np
    from mxnet_benchmarks.benchmarks import benchmark
    import json

    opts = {'model': 'resnet50', 'phase': 'training'}
    model_title, times = benchmark(opts)

    opts['results.time'] = np.mean(times)                                      # In seconds.
    opts['results.throughput'] = opts['batch_size'] / opts['results.time']     # Images / sec
    print(json.dumps(opts, indent=4))                                          # Prints benchmark details.

To list supported models, run the following code:

>>> from mxnet_benchmarks.model_factory import ModelFactory
>>> print(ModelFactory.models.keys())

To run from a command line, this module accepts the following parameters:

* **--model** A model to benchmark ("alexnet", "googlenet" ...)
* **--forward_only** Benchmark inference (if true) else benchmark training
* **--batch_size** Per device batch size
* **--num_warmup_batches** Number of warmup iterations
* **--num_batches** Number of benchmark iterations
* **--device** Comptue device, "cpu" or "gpu"
* **--num_gpus** Number of gpus to use (per node). Use CUDA_VISIBLE_DEVICES to select those devices
* **--data_dir** Path to the LMDB or LEVELDB data base
* **--kv_store** Type of gradient aggregation schema (local, device, dist_sync, dist_device_sync, dist_async).
      See https://mxnet.incubator.apache.org/how_to/multi_devices.html for more details.
* **--dtype** Precision of data variables: float(same as float32), float32 or float16
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import timeit
import json
import argparse
import traceback
import logging
import mxnet as mx
import numpy as np
from mxnet_benchmarks.data_iterator import DataIteratorFactory
from mxnet_benchmarks.model_factory import ModelFactory


def get_local_batch_size(opts):
    """Returns local batch size. This is an `effective` batch size for one node.

    Args:
        opts (dict): A dictionary containing benchmark parameters. Must contain `batch_size`, `device` and optionally
            `num_gpus`.

    Returns:
        Local batch size. Type int.
    """
    num_devices = 1 if opts['device'] == 'cpu' else opts['num_gpus']
    return opts['batch_size'] * num_devices


def get_devices(opts):
    """Creates MXNet device descriptions.

    Args:
        opts (dict): A dictionary containing benchmark parameters. Must contain `batch_size`, `device` and optionally
            `num_gpus`.

    Returns:
        List of devices (mx.cpu or mx.gpu)
    """
    if opts['device'] == 'cpu':
        devices = [mx.cpu()]
    else:
        devices = [mx.gpu(i) for i in range(opts['num_gpus'])]
    return devices


class BatchEndCallback(object):
    """A callback for an end-of-batch event. Counts number of batches processed.

    A non-standard implementation. The `__call__` method returns boolean value.
    False indicates training must stop.
    """
    def __init__(self, num_warmup_batches, num_batches):
        """Initializes this callback.

        :param int num_warmup_batches: Number of warmup batches.
        :param int num_batches: Number of benchmark batches.
        """
        self.num_warmup_batches = num_warmup_batches
        self.batches_done = 0
        self.batch_times = np.zeros(num_batches)
        self.tic = timeit.default_timer()

    def __call__(self, param):
        """Is called by `EarlyStoppableModule.fit`.

        :param mx.model.BatchEndParam param: Parameters.
        :return: False if training must be stopped.
        """
        batch_time = timeit.default_timer() - self.tic
        self.batches_done += 1
        idx = -1
        if self.batches_done > self.num_warmup_batches:
            idx = self.batches_done - self.num_warmup_batches - 1
            if not 0 <= idx < len(self.batch_times):
                raise ValueError("RuntimeError in BatchEndCallback::__call__[idx=%d, batches_done=%d, "
                                 "num_warmup_batches=%d]" % (idx, self.batches_done, self.num_warmup_batches))
            self.batch_times[idx] = batch_time
            # print("Batch %d completed in %f ms (array size %d)" % (self.batches_done,
            #                                                        1000.0*batch_time,
            #                                                        len(self.batch_times)))
        self.tic = timeit.default_timer()
        return False if idx == len(self.batch_times)-1 else True


def run_n_times(module, batch, opts):
    """ Run model **module** with input data **batch** as many times as specified in
        **opts**. Used for inference, not for training.

    Args:
        module (mxnet.mod.Module): MXNet model to use.
        batch (mx.io.DataBatch): One batch of data to use.
        opts (dict): Dictionary with options.

    Returns:
        Batch times (excluding warmup batches) in seconds. It is a numpy array. Length is opts['num_batches'].
    """
    print("[WARNING] This is ancient code and you do not really want to run inference tests with MXNET. Use "
          "TensorRT backend for that (-Pexp.framework='\"tensorrt\"')")
    is_train = opts['phase'] == 'training'
    if is_train is True:
        raise ValueError("This function must not be used in train phase.")
    for i in range(opts['num_warmup_batches']):
        module.forward(batch, is_train=is_train)
    batch_times = np.zeros(opts['num_batches'])
    for i in range(opts['num_batches']):
        start_time = timeit.default_timer()
        module.forward(batch, is_train=is_train)
        mx.nd.waitall()
        batch_times[i] = timeit.default_timer() - start_time
    return batch_times


def benchmark(opts):
    """Runs inference or training benchmarks depending on **opts['phase']** value.

    Args:
        opts (dict): Options for a benchmark. Must contain `model` and 'phase'. Other options are optional.

    Returns:
        Tuple of model title and numpy array containing batch times (string, numpy array).
    """
    for param in ['model', 'phase']:
        if param not in opts:
            raise ValueError("Missing '%s' in options" % param)
    if opts['phase'] not in ['inference', 'training']:
        raise ValueError("Invalid value for 'phase' (%s). Must be 'inference' or 'training'." % opts['phase'])
    try:
        opts['model_opts'] = json.loads(opts.get('model_opts', "{}"))
    except ValueError:
        raise ValueError("[ERROR] Cannot decode JSON string: '%s'" % opts['model_opts'])
    opts['batch_size'] = opts.get('batch_size', 16)
    opts['num_warmup_batches'] = opts.get('num_warmup_batches', 10)
    opts['num_batches'] = opts.get('num_batches', 10)
    opts['device'] = opts.get('device', 'gpu')
    opts['num_gpus'] = opts.get('num_gpus', 1)
    opts['dtype'] = opts.get('dtype', 'float')

    model = ModelFactory.get_model(opts)
    if opts['phase'] == 'inference':
        return benchmark_inference(model, opts)
    else:
        return benchmark_training(model, opts)


def benchmark_inference(model, opts):
    """ Runs N inferences and returns array of batch times in seconds.

    Args:
        model (obj): A model from `./models` folder.
        opts (dict): Options for the inference benchmark.

    Returns:
        Tuple of model title and numpy array containing batch times (string, numpy array).
    """
    if opts['device'] == 'gpu' and opts['num_gpus'] != 1:
        raise ValueError("When inference is performed on a GPU, only one GPU (--num_gpus=1) must be specified.")

    data_shape = [('data', (opts['batch_size'],) + model.input_shape)]
    device = get_devices(opts)[0]

    mod = mx.mod.Module(symbol=model.output, context=device, label_names=None)
    mod.bind(for_training=False, inputs_need_grad=False, data_shapes=data_shape)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    # TODO: Fix me here
    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=device) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, [])  # empty label

    return model.name, run_n_times(mod, batch, opts)


def benchmark_training(model, opts):
    """ Run training benchmarks.

    Args:
        model (obj): A model from `./models` folder.
        opts (dict): Options for the inference benchmark.

    Returns:
        Tuple of model title and numpy array containing batch times (string, numpy array).
    """
    # Label tensor always have shape (N,)
    kv = mx.kvstore.create(opts['kv_store'])
    # Create data iterator and resize it to total number of iterations (no matter what input data size is)
    train_data = DataIteratorFactory.get(
        (get_local_batch_size(opts),) + model.input_shape,
        (get_local_batch_size(opts),) + model.labels_shape,
        model.labels_range,
        opts,
        kv_store=kv
    )
    devices = get_devices(opts)
    optimizer_params = {'multi_precision': True} if opts['dtype'] == 'float16' else {}
    mod = mx.mod.Module(symbol=model.output, context=devices)
    batch_end_callback = BatchEndCallback(opts['num_warmup_batches'], opts['num_batches'])
    # print ("Starting benchmarks.")
    # TODO: In current implementation, number of epochs must always equal to 1. It is iterator responsibility to
    #       iterate the right number of batched - warm up plus benchmark batches.
    mod.fit(
        train_data,
        kvstore=kv,
        optimizer='sgd',
        optimizer_params=optimizer_params,
        eval_metric=model.eval_metric,
        initializer=mx.init.Normal(),
        batch_end_callback=[batch_end_callback],
        begin_epoch=0,
        num_epoch=1
    )
    # print ("Finished benchmarks.")
    return model.name, batch_end_callback.batch_times


def main():
    if os.environ.get('DLBS_DEBUG', '0') == '1':
        logging.getLogger().setLevel(logging.DEBUG)
    print("__mxnet.version__=%s" % (json.dumps(mx.__version__)))

    def str2bool(v):
        return v.lower() in ('true', 'on', 't', '1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, default='',
                        help="A model to benchmark ('alexnet', 'googlenet' ...)")
    parser.add_argument('--model_opts', type=str, required=False, default='{}',
                        help="Model's additional parameters (flat JSON dictionary).")
    parser.add_argument('--forward_only', nargs='?', const=True, default=False, type=str2bool,
                        help="Benchmark inference (if true) else benchmark training.")
    parser.add_argument('--batch_size', type=int, required=True, default=None, help="Per device batch size")
    parser.add_argument('--num_batches', type=int, required=False, default=100, help="Number of benchmark iterations")
    parser.add_argument('--num_warmup_batches', type=int, required=False, default=1, help="Number of warmup iterations")
    parser.add_argument('--num_gpus', type=int, required=False, default=1,
                        help="Number of gpus to use (per node?). Use CUDA_VISIBLE_DEVICES to select those devices.")
    parser.add_argument('--num_workers', type=int, required=False, default=1,
                        help="Number of workers participating in training.")
    parser.add_argument('--device', type=str, required=False, default='cpu', help="Compute device, 'cpu' or 'gpu'")
    parser.add_argument('--kv_store', type=str, required=False, default='device',
                        help="Type of gradient aggregation schema (local, device, dist_sync, dist_device_sync, "
                             "dist_async). See https://mxnet.incubator.apache.org/how_to/multi_devices.html "
                             "for more details.")
    parser.add_argument('--dtype', required=False, default='float', choices=['float', 'float32', 'float16'],
                        help="Precision of data variables: float(same as float32), float32 or float16.")
    parser.add_argument('--data_dir', type=str, required=False, default='',
                        help="Path to the image RecordIO (.rec) file or a directory path. "
                             "Created with tools/im2rec.py.")
    parser.add_argument('--preprocess_threads', type=int, required=False, default=4,
                        help="Number preprocess threads for data ingestion pipeline when real data is used.")
    parser.add_argument('--prefetch_buffer', type=int, required=False, default=10,
                        help="Number of batches to prefetch (buffer size)")
    parser.add_argument('--input_layout', type=str, required=False, default=None,
                        help="Layout of an input data tensor. It can be specified for certain cases, in particular, "
                             "for convolutional neural networks. Certain data iterators does not support all layouts. "
                             "Possible values: 'NCHW' and 'NHWC'. If not specified, default is 'NCHW'.")
    parser.add_argument('--model_layout', type=str, required=False, default=None,
                        help="Layout of tensors in a model compute graph. Can be specified for certain cases, like "
                             "convolutional neural networks when running them inside NGC mxnet containers which "
                             "provides extended functionality compared to standard mxnet distribution.  If not "
                             "specified, default is 'NCHW'.")
    parser.add_argument('--nvidia_layers', nargs='?', const=True, default=False, type=str2bool,
                        help="This is probably a temporary functionality. If set, it is assumed that this code runs in "
                             "NGC container with extended functionality. ")
    parser.add_argument('--use_dali', nargs='?', const=True, default=False, type=str2bool,
                        help="Use DALI for data ingestion pipeline.")
    # The workspace description is taken from here:
    #  https://mxnet.incubator.apache.org/api/python/symbol/symbol.html
    parser.add_argument('--workspace', type=int, required=False, default=1024,
                        help="Maximum temporary workspace allowed (MB) in convolution.This parameter has two usages. "
                             "When CUDNN is not used, it determines the effective batch size of the convolution "
                             "kernel. When CUDNN is used, it controls the maximum temporary storage used for tuning " 
                             "the best CUDNN kernel when limited_workspace strategy is used.")
    args = parser.parse_args()

    if args.dtype == 'float':
        args.dtype = 'float32'

    try:
        opts = vars(args)
        opts['phase'] = 'inference' if args.forward_only else 'training'
        model_title, times = benchmark(opts)
    except Exception as e:
        # TODO: this is not happening, program terminates earlier.
        # For now, do not rely on __results.status__=...
        times = np.zeros(0)
        opts = {}
        model_title = 'Unk'
        print("Critical error while running benchmarks (%s). See stacktrace below." % (str(e)))
        traceback.print_exc(file=sys.stdout)

    if len(times) > 0:
        mean_time = np.mean(times)                                                      # seconds
        mean_throughput = opts['num_workers'] * get_local_batch_size(opts) / mean_time  # images / sec
        print("__results.time__=%s" % (json.dumps(1000.0 * mean_time)))
        print("__results.throughput__=%s" % (json.dumps(int(mean_throughput))))
        print("__exp.model_title__=%s" % (json.dumps(model_title)))
        print("__results.time_data__=%s" % (json.dumps((1000.0*times).tolist())))
    else:
        print("__results.status__=%s" % (json.dumps("failure")))
    # Need this because of os._exit below to make sure that all gets printed.
    sys.stdout.flush()
    sys.stderr.flush()
    # TODO: Without exit call mxnet seems to hang in distributed mode.
    #    https://stackoverflow.com/questions/73663/terminating-a-python-script
    #    https://stackoverflow.com/a/5120178/1278994
    os._exit(os.EX_OK)


if __name__ == '__main__':
    main()
