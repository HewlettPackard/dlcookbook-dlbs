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

Horovod:
    https://github.com/apache/incubator-mxnet/blob/master/example/distributed_training-horovod/README.md
    https://github.com/apache/incubator-mxnet/blob/master/example/distributed_training-horovod/resnet50_imagenet.py
    https://github.com/horovod/horovod/blob/master/docs/mpirun.rst
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import timeit
import json
import argparse
import logging
import re
import mxnet as mx
import numpy as np
from mxnet_benchmarks.data_iterator import DataIteratorFactory
from mxnet_benchmarks.model_factory import ModelFactory
try:
    # https://github.com/apache/incubator-mxnet/blob/master/example/distributed_training-horovod/README.md
    import horovod.mxnet as hvd
except ImportError:
    hvd = None


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


class Benchmark(object):
    def __init__(self, args):
        self.args = args
        self.is_inference = args.phase == 'inference'
        self.is_horovod = 'horovod' in args.kv_store
        self.is_gpu = len(args.gpus) > 0
        self.kv_store = None if self.is_inference or self.is_horovod else mx.kvstore.create(args.kv_store)
        self.devices = [mx.cpu()] if not self.is_gpu else [mx.gpu(i) for i in self.args.gpus]
        self.model = ModelFactory.get_model(vars(args))
        #
        self.worker_batch = args.batch_size
        if 'horovod' not in args.kv_store and self.is_gpu:
            self.worker_batch *= len(args.gpus)
        # TODO: Is this implemented in NGC 19-05 in kv_store? There's a test below. I test it here because
        #       data iterators depend on rank and world size as returned by kv_store.
        #       Update - this does not work. KVStore reports the wrong number of workers/rank with Horovod.
        if self.is_horovod:
            self.num_workers = hvd.size()
            self.rank = hvd.rank()
        else:
            self.num_workers = 1 if self.kv_store is None else self.kv_store.num_workers
            self.rank = 0 if self.kv_store is None else self.kv_store.rank
        #
        self.effectve_batch = self.worker_batch * self.num_workers
        logging.info("is_horovod=%s, num_workers=%d, rank=%d, worker_batch=%d, effective_batch=%d",
                     self.is_horovod, self.num_workers, self.rank, self.worker_batch, self.effectve_batch)

    def run(self):
        return self.run_inference() if self.is_inference else self.run_training()

    def run_inference(self):
        """ Runs N inferences and returns array of batch times in seconds.

        Returns:
            Numpy array containing batch times (string, numpy array).
        """
        logging.warning("[WARNING] This is ancient code and you do not really want to run inference tests with MXNET. "
                        "Use TensorRT backend for that (-Pexp.framework='\"tensorrt\"')")

        if len(self.devices) > 1:
            raise ValueError("Multiple devices ({}) are not supported in inference phase.".format(str(self.devices)))

        data_shape = [('data', (self.worker_batch,) + self.model.input_shape)]
        mod = mx.mod.Module(symbol=self.model.output, context=self.devices[0], label_names=[])
        mod.bind(for_training=False, inputs_need_grad=False, data_shapes=data_shape)
        mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
        # TODO: Fix me here
        data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=self.devices[0]) for _, shape in mod.data_shapes]
        batch = mx.io.DataBatch(data, [])  # empty label

        for i in range(self.args.num_warmup_batches):
            mod.forward(batch, is_train=False)
        batch_times = np.zeros(self.args.num_batches)
        for i in range(self.args.num_batches):
            start_time = timeit.default_timer()
            mod.forward(batch, is_train=False)
            mx.nd.waitall()
            batch_times[i] = timeit.default_timer() - start_time

        return batch_times

    def run_training(self):
        """ Run training benchmarks.
        Returns:
            Numpy array containing batch times (string, numpy array).
        """
        # Create data iterator and resize it to total number of iterations (no matter what input data size is)
        train_data = DataIteratorFactory.get((self.worker_batch,) + self.model.input_shape,
                                             (self.worker_batch,) + self.model.labels_shape,
                                             self.model.labels_range,
                                             self.args,
                                             kv_store=self.kv_store)
        # https://github.com/apache/incubator-mxnet/blob/master/example/distributed_training-horovod/resnet50_imagenet.py
        optimizer_params = {'multi_precision': True} if self.args.dtype == 'float16' else {}
        if self.is_horovod:
            optimizer_params['rescale_grad'] = 1.0 / self.worker_batch
        opt = mx.optimizer.create('sgd', **optimizer_params)
        if self.is_horovod:
            opt = hvd.DistributedOptimizer(opt)

        mod = mx.mod.Module(symbol=self.model.output, context=self.devices[0])
        mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label, for_training=True)
        mod.init_params(mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
        if self.is_horovod:
            arg_params, aux_params = mod.get_params()
            if arg_params:
                hvd.broadcast_parameters(arg_params, root_rank=0)
            if aux_params:
                hvd.broadcast_parameters(aux_params, root_rank=0)
            mod.set_params(arg_params=arg_params, aux_params=aux_params)

        batch_end_callback = BatchEndCallback(self.args.num_warmup_batches, self.args.num_batches)
        # print ("Starting benchmarks.")
        # TODO: In current implementation, number of epochs must always equal to 1. It is iterator responsibility to
        #       iterate the right number of batched - warm up plus benchmark batches.
        mod.fit(train_data,
                kvstore=self.kv_store,
                optimizer=opt,
                optimizer_params=optimizer_params,
                eval_metric=self.model.eval_metric,
                batch_end_callback=[batch_end_callback],
                begin_epoch=0,
                num_epoch=1)

        if self.is_horovod:
            start_time = timeit.default_timer()
            mx.ndarray.waitall()
            logging.info("(horovod) wait time for all ndarrays is %.5f seconds", timeit.default_timer() - start_time)
        return batch_end_callback.batch_times


def main():
    """
        Environmental variables:
            DLBS_MXNET_SYNTHETIC_DATA_CONTEXT  [cpu]
            DLBS_MXNET_LOG_LEVEL               [INFO]
    """
    logging.basicConfig()
    logging.getLogger().setLevel(os.environ.get('DLBS_MXNET_LOG_LEVEL', 'INFO'))

    def str2bool(v):
        return v.lower() in ('true', 'on', 't', '1')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, default='',
                        help="A model to benchmark ('alexnet', 'googlenet' ...)")
    parser.add_argument('--model_opts', type=json.loads, required=False, default={},
                        help="Model's additional parameters (flat JSON dictionary).")
    parser.add_argument('--phase', required=False, default='training', choices=['inference', 'training'],
                        help="Benchmark phase - training or inference.")
    parser.add_argument('--batch_size', type=int, required=True, default=None, help="Replica (per device) batch size.")
    parser.add_argument('--num_batches', type=int, required=False, default=100, help="Number of benchmark iterations")
    parser.add_argument('--num_warmup_batches', type=int, required=False, default=1, help="Number of warmup iterations")
    parser.add_argument('--gpus', type=str, required=False, default='0',
                        help="Comma-separated list of GPUs to use or empty for CPUs.")
    parser.add_argument('--kv_store', type=str, required=False, default='device',
                        help="Type of gradient aggregation schema (local, device, dist_sync, dist_device_sync, "
                             "dist_async). See https://mxnet.incubator.apache.org/how_to/multi_devices.html "
                             "for more details.")
    parser.add_argument('--dtype', required=False, default='float', choices=['float32', 'float16'],
                        help="Precision of data variables: float32 or float16.")
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
    if 'horovod' in args.kv_store:
        if hvd:
            hvd.init()
            if hvd.local_rank() != 0:
                logging.getLogger().setLevel(logging.ERROR)
        else:
            raise ValueError("Horovod library not found")

    args.gpus = [int(gpu) for gpu in re.sub('[:,]', ' ', args.gpus).split()]
    if 'horovod' in args.kv_store and len(args.gpus) > 0:
        args.gpus = [args.gpus[hvd.local_rank()]]

    benchmark = Benchmark(args)
    if benchmark.rank == 0:
        print("__mxnet.version__=%s" % (json.dumps(mx.__version__)))
        print("__exp.framework_ver__=%s" % (json.dumps(mx.__version__)))

    try:
        batch_times = benchmark.run()
    except Exception:
        # TODO: this is not happening, program terminates earlier.
        # For now, do not rely on __results.status__=...
        batch_times = np.zeros(0)
        logging.exception("Critical error while running benchmarks. See stacktrace below.")

    if benchmark.rank == 0:
        if len(batch_times) > 0:
            mean_time = np.mean(batch_times)  # -----------------------------  in seconds
            mean_throughput = benchmark.effectve_batch / mean_time  # -------  images / sec
            print("__results.time__=%s" % (json.dumps(1000.0 * mean_time)))  # in milliseconds
            print("__results.throughput__=%s" % (json.dumps(int(mean_throughput))))
            print("__exp.model_title__=%s" % (json.dumps(benchmark.model.name)))
            print("__results.time_data__=%s" % (json.dumps((1000.0*batch_times).tolist())))
        else:
            print("__results.status__=%s" % (json.dumps("failure")))
    # Need this because of os._exit below to make sure that all gets printed.
    sys.stdout.flush()
    sys.stderr.flush()
    # TODO: Without exit call mxnet seems to hang in distributed mode.
    #    https://stackoverflow.com/questions/73663/terminating-a-python-script
    #    https://stackoverflow.com/a/5120178/1278994
    if not benchmark.is_horovod:
        os._exit(os.EX_OK)


if __name__ == '__main__':
    main()
