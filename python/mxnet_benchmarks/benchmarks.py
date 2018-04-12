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
    Based on this Image Classification example: https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/
    TODO: Think about using that code to do benchmarking?

This module can run from command line or from python application. To benchmark a particular
model on a synthetic data, run the following commands:

>>> import numpy as np
>>> from mxnet_benchmarks.benchmarks import benchmark
>>> import json
>>>
>>> opts = {'model': 'resnet50', 'phase': 'training'}
>>> model_title, times = benchmark(opts)
>>>
>>> opts['results.time'] = np.mean(times)                                      # In seconds.
>>> opts['results.throughput'] = opts['batch_size'] / opts['results.time']     # Images / sec
>>> print(json.dumps(opts, indent=4))                                          # Prints benchmark details. Parameters can be overriden.

To list supported models, run the following code:

>>> from mnet_benchmarks.model_factory import ModelFactory
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
* **--kv_store** Type of gradient aggregation schema (local, device, dist_sync, dist_device_sync, dist_async). See https://mxnet.incubator.apache.org/how_to/multi_devices.html for more details.
* **--dtype** Precision of data variables: float(same as float32), float32 or float16
"""

from __future__ import absolute_import
from __future__ import print_function
import sys
import timeit
import time
import json
import argparse
import traceback
import mxnet as mx
import numpy as np
from mxnet_benchmarks.data_iterator import DataIteratorFactory
from mxnet_benchmarks.model_factory import ModelFactory


def get_effective_batch_size(opts):
    """Returns effective batch size

    :param dict opts: A dictionary containing benchmark parameters. Must contain
                      `batch_size`, `device` and optionally `num_gpus`.
    :return: Effective batch size.
    :rtype: int
    """
    num_devices = 1 if opts['device'] == 'cpu' else opts['num_gpus']
    return opts['batch_size'] * num_devices


def get_devices(opts):
    """Creates MXNet device descriptions.

    :param dict opts: A dictionary containing benchmark parameters. Must contain
                      `batch_size`, `device` and optionally `num_gpus`.
    :return: List of devices (mx.cpu or mx.gpu)
    """
    devs = None
    if opts['device'] == 'cpu':
        devs = [mx.cpu()]
    else:
        devs = [mx.gpu(i) for i in range(opts['num_gpus'])]
    return devs


class BenchmarkingModule(mx.mod.Module):
    """This is a copy past from mxnet project. 
    
    I think, the source file is https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/module/base_module.py
    The reason to have it here is to be able to perform predefined number of iterations 
    that include warmup and benchmark iterations. There are two major differences 
    compared to mxnet implementation:
    
    1. Numer of epochs is ignored
    2. BatchEndCallback returns False indicating that training must be stopped.
    """

    def __init__(self, *args, **kwargs):
        super(BenchmarkingModule, self).__init__(*args, **kwargs)

    def fit(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_end_callback=None,
            eval_batch_end_callback=None, initializer=mx.initializer.Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None):

        #assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
                  for_training=True, force_rebind=force_rebind)
        if monitor is not None:
            self.install_monitor(monitor)
        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)

        if validation_metric is None:
            validation_metric = eval_metric
        if not isinstance(eval_metric, mx.metric.EvalMetric):
            eval_metric = mx.metric.create(eval_metric)

        ################################################################################
        # training loop
        ################################################################################
        early_stop = False
        epoch = -1
        while True:
            epoch = epoch + 1
            tic = time.time()
            eval_metric.reset()
            nbatch = 0
            data_iter = iter(train_data)
            end_of_batch = False
            next_data_batch = next(data_iter)
            while not end_of_batch:
                data_batch = next_data_batch
                if monitor is not None:
                    monitor.tic()
                self.forward_backward(data_batch)
                self.update()
                #mx.nd.waitall()
                try:
                    # pre fetch next batch
                    next_data_batch = next(data_iter)
                    self.prepare(next_data_batch)
                except StopIteration:
                    end_of_batch = True
                #print(self._exec_group.labels_.dtype)  
                self.update_metric(eval_metric, data_batch.label)

                if monitor is not None:
                    monitor.toc_print()

                if batch_end_callback is not None:
                    batch_end_params = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                              eval_metric=eval_metric,
                                                              locals=locals())
                    for callback in mx.base._as_list(batch_end_callback):
                        if not callback(batch_end_params):
                            early_stop = True
                nbatch += 1
                if early_stop:
                    break

            # one epoch of training is finished
            for name, val in eval_metric.get_name_value():
                self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            # sync aux params across devices
            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)

            if epoch_end_callback is not None:
                for callback in mx.base._as_list(epoch_end_callback):
                    callback(epoch, self.symbol, arg_params, aux_params)

            #----------------------------------------
            # evaluation on validation set
            if eval_data:
                res = self.score(eval_data, validation_metric,
                                 score_end_callback=eval_end_callback,
                                 batch_end_callback=eval_batch_end_callback, epoch=epoch)
                #TODO: pull this into default
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

            # end of 1 epoch, reset the data-iter for another epoch
            if early_stop:
                break
            train_data.reset()


class BatchEndCallback(object):
    """A callback for an end-of-batch event. Counts number of batches processed.

    A non-standard implementation. The `__call__` method returns boolean value.
    False indicates training must stop.
    """
    def __init__(self, num_warmup_batches, num_batches):
        """Initialzies this callback.

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
            assert idx >= 0 and idx < len(self.batch_times), "RuntimeError in BatchEndCallback::__call__[idx=%d, batches_done=%d, num_warmup_batches=%d]" % (idx, self.batches_done, self.num_warmup_batches)
            self.batch_times[idx] = batch_time
            #print ("Batch %d completed in %f ms (array size %d)" % (self.batches_done, 1000.0*batch_time, len(self.batch_times)))
        self.tic = timeit.default_timer()
        return False if idx == len(self.batch_times)-1 else True


def run_n_times(module, batch, opts):
    """ Run model **module** with input data **batch** as many times as specified in
        **opts**. Used for inference, not for training.

    :param mxnet.mod.Module module: MXNet model to use
    :param batch: List of mxnet.ndarray
    :return: Batch times (excluding warmup batches) in seconds.
    :rtype: Numpy array of length = **num_batches**.
    """
    is_train = opts['phase'] == 'training'
    assert is_train == False, "This function must not be used in train phase."
    for i in range(opts['num_warmup_batches']):
        module.forward(batch, is_train=is_train)
        if is_train:
            module.backward()
            module.update()
    batch_times = np.zeros(opts['num_batches'])
    for i in range(opts['num_batches']):
        start_time = timeit.default_timer()
        module.forward(batch, is_train=is_train)
        if is_train:
            module.backward()
            module.update()
        mx.nd.waitall()
        batch_times[i] = timeit.default_timer() - start_time
    return batch_times


def benchmark(opts):
    """Runs inference or training benchmarks depending on **opts['phase']** value.

    :param dict opts: Options for a benchmark. Must contain `model` and 'phase'.\
                      Other options are optional.
    :return: Tuple of model title and numpy array containing batch times.
    :rtype: (string, numpy array)
    """
    assert 'model' in opts, "Missing 'model' in options."
    assert 'phase' in opts, "Missing 'phase' in options."
    assert opts['phase'] in ['inference', 'training'], "Invalid value for 'phase' (%s). Must be 'inference' or 'training'." % (opts['phase'])
    
    opts['batch_size'] = opts.get('batch_size', 16)
    opts['num_warmup_batches'] = opts.get('num_warmup_batches', 10)
    opts['num_batches'] = opts.get('num_batches', 10)
    opts['device'] = opts.get('device', 'gpu')
    opts['num_gpus'] = opts.get('num_gpus', 1)
    opts['dtype'] = opts.get('dtype', 'float')
    opts['enable_tensor_core'] = opts.get('enable_tensor_core', False)

    model = ModelFactory.get_model(opts)
    if opts['phase'] == 'inference':
        return benchmark_inference(model, opts)
    else:
        return benchmark_training(model, opts)


def benchmark_inference(model, opts):
    """ Runs N inferences and returns array of batch times in seconds.

    :param obj model: A model from `./models` folder.
    :param dict opts: Options for the inference benchmark.
    :return: Tuple of model title and numpy array containing batch times.
    :rtype: (string, numpy array)
    """
    if opts['device'] == 'gpu':
        assert opts['num_gpus'] == 1, "When inference is performed on a GPU, only one GPU (--num_gpus=1) must be specified."

    data_shape = [('data', (opts['batch_size'],) + model.input_shape)]
    device = get_devices(opts)[0]

    mod = mx.mod.Module(symbol=model.output, context=device, label_names=None)
    mod.bind(for_training=False, inputs_need_grad=False, data_shapes=data_shape)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

    data = [mx.random.uniform(-1.0, 1.0, shape=shape, ctx=device) for _, shape in mod.data_shapes]
    batch = mx.io.DataBatch(data, []) # empty label

    return (model.name, run_n_times(mod, batch, opts))


def benchmark_training(model, opts):
    """ Creates engine that runs training. """
    kv = mx.kvstore.create(opts['kv_store'])
    train_data = DataIteratorFactory.get(
        model.num_classes,
        (get_effective_batch_size(opts),) + model.input_shape,
        opts,
        kv
    )
    devices = get_devices(opts)

    mod = BenchmarkingModule(symbol=model.output, context=devices)
    batch_end_callback = BatchEndCallback(opts['num_warmup_batches'], opts['num_batches'])
    #print ("Starting benchmarks.")
    mod.fit(
        train_data,
        kvstore=kv,
        optimizer='sgd',
        optimizer_params = {'multi_precision': True},
        initializer=mx.init.Normal(),
        batch_end_callback=[batch_end_callback]
    )
    #print ("Finished benchmarks.")
    return (model.name, batch_end_callback.batch_times)


if __name__ == '__main__':
    # --model, --forward_only, -batch_size, --num_batches, --num_warmup_batches, --num_gpus, --device, --data_dir
    # --kv_store
    print("__mxnet.version__=%s" % (json.dumps(mx.__version__)))

    def str2bool(v):
        return v.lower() in ('true', 'on', 't', '1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, default='', help='A model to benchmark ("alexnet", "googlenet" ...)')
    parser.add_argument('--forward_only', nargs='?', const=True, default=False, type=str2bool, help='Benchmark inference (if true) else benchmark training.')
    parser.add_argument('--batch_size', type=int, required=True, default=None, help='Per device batch size')
    parser.add_argument('--num_batches', type=int, required=False, default=100, help='Number of benchmark iterations')
    parser.add_argument('--num_warmup_batches', type=int, required=False, default=1, help='Number of warmup iterations')
    parser.add_argument('--num_gpus', type=int, required=False, default=1, help='Number of gpus to use (per node?). Use CUDA_VISIBLE_DEVICES to select those devices.')
    parser.add_argument('--device', type=str, required=False, default='cpu', help='Comptue device, "cpu" or "gpu"')
    parser.add_argument('--kv_store', type=str, required=False, default='device', help='Type of gradient aggregation schema (local, device, dist_sync, dist_device_sync, dist_async).\
                                                                                        See https://mxnet.incubator.apache.org/how_to/multi_devices.html for more details.')
    parser.add_argument('--dtype', required=False, default='float', choices=['float', 'float32', 'float16'], help='Precision of data variables: float(same as float32), float32 or float16.')
    parser.add_argument('--data_dir', type=str, required=False, default='', help='Path to the image RecordIO (.rec) file or a directory path. Created with tools/im2rec.py.')

    parser.add_argument('--preprocess_threads', type=int, required=False, default=4, help='Number preprocess threads for data ingestion pipeline when real data is used.')
    parser.add_argument('--prefetch_buffer', type=int, required=False, default=10, help='Number of batches to prefetch (buffer size)')
    args = parser.parse_args()

    if args.dtype == 'float':
        args.dtype = 'float32'

    try:
        opts = vars(args)
        opts['phase'] = 'inference' if args.forward_only else 'training'
        model_title, times = benchmark(opts)
    except Exception, e:
        #TODO: this is not happenning, program terminates earlier.
        # For now, do not rely on __results.status__=...
        times = np.zeros(0)
        model_title = 'Unk'
        print ("Critical error while running benchmarks (%s). See stacktrace below." % (str(e)))
        traceback.print_exc(file=sys.stdout)

    if len(times) > 0:
        mean_time = np.mean(times)                                   # seconds
        mean_throughput = get_effective_batch_size(opts) / mean_time # images / sec
        print("__results.time__=%s" % (json.dumps(1000.0 * mean_time)))
        print("__results.throughput__=%s" % (json.dumps(int(mean_throughput))))
        print("__exp.model_title__=%s" % (json.dumps(model_title)))
        print("__results.time_data__=%s" % (json.dumps((1000.0*times).tolist())))
    else:
        print("__results.status__=%s" % (json.dumps("failure")))
