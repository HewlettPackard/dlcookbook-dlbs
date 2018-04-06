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
"""Caffe2 benchmarks entry point module.

This module can run from command line or from python application. To benchmark a particular
model on a synthetic data, run the following commands:

>>> import numpy as np
>>> from caffe2_benchmarks.benchmarks import benchmark
>>> import json
>>>
>>> opts = {'model': 'resnet50', 'phase': 'training'}
>>> model_title, times = benchmark(opts)
>>>
>>> opts['results.time'] = np.mean(times)                                      # In seconds.
>>> opts['results.throughput'] = opts['batch_size'] / opts['results.time']     # Images / sec
>>> print(json.dumps(opts, indent=4))                                          # Prints benchmark details. Parameters can be overriden.

To list supported models, run the following code:

>>> from caffe2_benchmarks.model_factory import ModelFactory
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
* **--data_backend** Database backend. One of "lmdb" or "leveldb"
* **--dtype** Precision of data variables: float(same as float32), float32 or float16
* **--enable_tensor_core** Enable volta\'s tensor ops (requires CUDA >= 9, cuDNN >= 7 and NVIDIA Volta GPU)
"""
from __future__ import absolute_import
from __future__ import print_function
import sys
import json
import timeit
import argparse
import traceback
import numpy as np
from caffe2.python import workspace
from caffe2.python import core
from caffe2.python import brew
from caffe2.python import model_helper
from caffe2.python import optimizer
from caffe2.python import data_parallel_model as dpm
from caffe2.python.modeling.initializers import Initializer, pFP16Initializer
from caffe2_benchmarks.models.model import Model
from caffe2_benchmarks.model_factory import ModelFactory


def run_n_times(model, num_warmup_batches, num_batches):
    """ Runs **model** multiple times (**num_warmup_batches** + **num_batches**).

    :param model: Caffe2's model helper class instances.
    :type model: :py:class:`caffe2.python.model_helper.ModelHelper`
    :param int num_warmup_batches: Number of warmup batches to process (do not contribute to computing average batch time)
    :param int num_batches: Number of batches to process (contribute to computing average batch time)
    :return: Batch times (excluding warmup batches) in seconds.
    :rtype: Numpy array of length = **num_batches**.
    """
    net_name = model.net.Proto().name
    start_time = timeit.default_timer()
    if num_warmup_batches > 0:
        workspace.RunNet(net_name, num_iter=num_warmup_batches)
        print("Average warmup batch time %f ms across %d batches" % (1000.0*(timeit.default_timer() - start_time)/num_warmup_batches, num_warmup_batches))
    else:
        print("Warning - no warmup iterations has been performed.")

    batch_times = np.zeros(num_batches)
    for i in range(num_batches):
        start_time = timeit.default_timer()
        workspace.RunNet(net_name, 1)
        batch_times[i] = timeit.default_timer() - start_time
    return batch_times


def create_model(model_builder, model, enable_tensor_core, float16_compute, loss_scale=1.0):
    """Creates one model replica.

    :param obj model_builder: A model instance that contains `forward_pass_builder` method.
    :param model: Caffe2's model helper class instances.
    :type model: :py:class:`caffe2.python.model_helper.ModelHelper`
    :param bool enable_tensor_core: If true, Volta's tensor core ops are enabled.
    :param float loss_scale: Scale loss for multi-GPU training.
    :return: Head nodes (softmax or loss depending on phase)
    """
    initializer = (pFP16Initializer if model_builder.dtype == 'float16' else Initializer)
    with brew.arg_scope([brew.conv, brew.fc],
                        WeightInitializer=initializer,
                        BiasInitializer=initializer,
                        enable_tensor_core=enable_tensor_core,
                        float16_compute=float16_compute):
        outputs = model_builder.forward_pass_builder(model, loss_scale=loss_scale)
    return outputs


def build_optimizer(model, float16_compute = False):
    if False: # float16_compute:   # A newwer versions of Caffe support this
        print("[INFO] Building FP16 SGD optimizer.")
        opt = optimizer.build_fp16_sgd(
            model, 0.1, momentum=0.9, policy='step', gamma=0.1, weight_decay=1e-4
        )
    else:
        print("[INFO] Building Multi-precision SGD optimizer.")
        optimizer.add_weight_decay(model, 1e-4)
        #opt = optimizer.build_sgd(
        opt = optimizer.build_multi_precision_sgd(
            model, 0.1, momentum=0.9, policy='fixed', gamma=0.1
        )
    return opt


def benchmark(opts):
    """Runs inference or training benchmarks depending on **opts['phase']** value.
    
    You may want to call **workspace.ResetWorkspace()** to clear everything once
    this method has exited.
    
    :param dict opts: Options for a benchmark. Must contain `model` and 'phase'.\
                      Other options are optional.
    :return: Tuple of model title and numpy array containing batch times.
    :rtype: (string, numpy array)
    
    Usage example:

    >>> opts = {'model': 'resnet50', 'phase': 'training'}
    >>> model_title, times = benchmark(opts)
    
    This function checks that **opts** contains all mandatory parameters, sets
    optional parameters to default values and depending on **phase** value,
    calls either :py:func:`benchmark_inference` or :py:func:`benchmark_training`.
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
    opts['num_decode_threads'] = opts.get('num_decode_threads', 1)
    opts['float16_compute'] = opts.get('float16_compute', False)
    
    if opts['device'] == 'gpu':
        print("[INFO] Creating ModelHelper for GPU. Optimizations are applied.")
        arg_scope = {
            'order': 'NCHW',
            'use_cudnn': opts['use_cudnn'],
            'cudnn_exhaustive_search': opts['cudnn_exhaustive_search'],
            'ws_nbytes_limit': (opts['cudnn_workspace_limit_mb'] * 1024 * 1024)
        }
        model = model_helper.ModelHelper(name=opts['model'], arg_scope=arg_scope)
    else:
        print("[WARNING] Creating ModelHelper for CPU. TODO: Apply similar optimziations as for GPUs.")
        model = model_helper.ModelHelper(name=opts['model'])
    if opts['phase'] == 'inference':
        return benchmark_inference(model, opts)
    else:
        return benchmark_training(model, opts)
    

def benchmark_inference(model, opts):
    """ Runs N inferences and returns array of batch times in seconds.

    :param model: Caffe2's model helper class instances.
    :type model: :py:class:`caffe2.python.model_helper.ModelHelper`
    :param dict opts: Options for the inference benchmark. Must contain `device`,\
                      `num_gpus` if device is gpu, `enable_tensor_core`,\
                      `num_warmup_batches` and `num_batches`. Optional parameters are\
                      `data_dir` and `data_backend`.
    :return: Tuple of model title and numpy array containing batch times.
    :rtype: (string, numpy array)
    """
    if opts['device'] == 'gpu':
        assert opts['num_gpus'] == 1,\
        "When inference is performed on a GPU, only one GPU (--num_gpus=1) must be specified."
    dev_opt = Model.get_device_option(0 if opts['device'] == 'gpu' else None)
    model_builder = ModelFactory.get_model(opts)
    with core.DeviceScope(dev_opt):
        create_model(model_builder, model, opts['enable_tensor_core'])
        model_builder.add_synthetic_inputs(model, add_labels=False)
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)
    return (model_builder.name, run_n_times(model, opts['num_warmup_batches'], opts['num_batches']))


def benchmark_training(model, opts):
    """ Runs N training batches and returns array of batch times in seconds.

    For some impl details see https://caffe2.ai/docs/SynchronousSGD.html.

    :param model: Caffe2's model helper class instances.
    :type model: :py:class:`caffe2.python.model_helper.ModelHelper`
    :param dict opts: Options for the inference benchmark. Must contain `device`,\
                      `num_gpus` if device is gpu, `enable_tensor_core`,\
                      `num_warmup_batches` and `num_batches`.
    :return: Tuple of model title and numpy array containing batch times.
    :rtype: (string, numpy array)
    """
    model_builder = ModelFactory.get_model(opts)
    assert model_builder.phase == 'training',\
           "Internal error, invalid phase was set. "\
           "Expecting 'training' but found %s" % (model_builder.phase)

    # Reader must be shared by all GPUs in a one machine.
    reader = None
    if 'data_dir' in opts and opts['data_dir']:
        reader = model.CreateDB(
            "reader",
            db=opts['data_dir'],            # (str, path to training data)
            db_type=opts['data_backend'],   # (str, 'lmdb' or 'leveldb')
            num_shards=1,                   # (int, number of machines)
            shard_id=0,                     # (int, machine id)
        )

    def add_inputs(model):
        if reader is None:
            print("[INFO] Adding synthetic data input for Caffe2 training benchmarks")
            model_builder.add_synthetic_inputs(model, add_labels=True)
        else:
            print("[INFO] Adding real data inputs (%s) for Caffe2 training benchmarks" % (opts['data_dir']))
            model_builder.add_data_inputs(
                model, reader, use_gpu_transform=(opts['device'] == 'gpu'),
                num_decode_threads = opts['num_decode_threads']
            )

    def create_net(model, loss_scale):
        return create_model(model_builder, model, opts['enable_tensor_core'],
                            opts['float16_compute'], loss_scale)

    def add_post_sync_ops(model):
        """Add ops applied after initial parameter sync."""
        for param_info in model.GetOptimizationParamInfo(model.GetParams()):
            if param_info.blob_copy is not None:
                model.param_init_net.HalfToFloat(
                    param_info.blob,
                    param_info.blob_copy[core.DataType.FLOAT]
                )
    def add_optimizer(model):
        return build_optimizer(model, float16_compute=opts['float16_compute'])

    if opts['device'] == 'gpu':
        dpm.Parallelize(
            model,
            input_builder_fun=add_inputs,
            forward_pass_builder_fun=create_net,
            optimizer_builder_fun=add_optimizer,
            #param_update_builder_fun=Model.add_parameter_update_ops,
            post_sync_builder_fun=add_post_sync_ops,
            devices=range(opts['num_gpus']),
            optimize_gradient_memory=True,
            cpu_device=(opts['device'] == 'cpu'),
            shared_model=(opts['device'] == 'cpu')
        )
    else:
        with core.DeviceScope(Model.get_device_option(gpu=None)):
            add_inputs(model)
            losses = create_net(model, 1.0)
            blobs_to_gradients = model.AddGradientOperators(losses)
            Model.add_parameter_update_ops(model)
        Model.optimize_gradient_memory(model, [blobs_to_gradients[losses[0]]])

    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)
    return (model_builder.name, run_n_times(model, opts['num_warmup_batches'], opts['num_batches']))


if __name__ == '__main__':
    # --model, --forward_only, -batch_size, --num_batches, --num_warmup_batches,
    # --num_gpus, --device, --data_dir, --kv_store
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
    parser.add_argument('--data_dir', type=str, required=False, default='', help='Path to the LMDB or LEVELDB data base.')
    parser.add_argument('--data_backend', required=False, default='lmdb', choices=['lmdb', 'leveldb'], help='Database backend. One of "lmdb" or "leveldb".')
    parser.add_argument('--dtype', required=False, default='float', choices=['float', 'float32', 'float16'], help='Precision of data variables: float(same as float32), float32 or float16.')
    parser.add_argument('--enable_tensor_core', action='store_true', help='Enable volta\'s tensor ops (requires CUDA >= 9, cuDNN >= 7 and NVIDIA Volta GPU)')
    parser.add_argument('--num_decode_threads', type=int, required=False, default=1, help='Number of image decode threads. For high throughput models such as AlexNetOWT set to 6-8 for 4 Voltas.')
    parser.add_argument('--float16_compute', nargs='?', const=True, default=False, type=str2bool, help='If true, use FP16 SGD optimizer else use multi-precision SGD optimizer')
    # These parameters affect the ModelHelper behaviour and are now applied for GPU benchmarks
    parser.add_argument('--cudnn_workspace_limit_mb', type=int, required=False, default=64, help='CuDNN workspace limit in MBs')
    parser.add_argument('--use_cudnn', nargs='?', const=True, default=True, type=str2bool, help='Use NVIDIA cuDNN library.')
    parser.add_argument('--cudnn_exhaustive_search', nargs='?', const=True, default=True, type=str2bool, help='Benchmark inference (if true) else benchmark training.')
    args = parser.parse_args()

    if args.dtype == 'float32':
        args.dtype = 'float'

    # report some available info
    if args.device == 'gpu':
        assert args.num_gpus > 0, "Number of GPUs must be specified in GPU mode"
        print("__caffe2.cuda_version__=%s" % (json.dumps(workspace.GetCUDAVersion())))
        print("__caffe2.cudnn_version__=%s" % (json.dumps(workspace.GetCuDNNVersion())))

    try:        
        opts = vars(args)
        opts['phase'] = 'inference' if args.forward_only else 'training'
        model_title, times = benchmark(opts)
    except Exception as err:
        #TODO: this is not happenning, program terminates earlier.
        # For now, do not rely on __results.status__=...
        times = np.zeros(0)
        model_title = 'Unk'
        print ("Critical error while running benchmarks (%s). See stacktrace below." % (str(err)))
        traceback.print_exc(file=sys.stdout)

    if len(times) > 0:
        mean_time = np.mean(times)                        # seconds
        mean_throughput = args.batch_size / mean_time     # images / sec
        if args.device == 'gpu':
            mean_throughput = mean_throughput * args.num_gpus
        print("__results.time__=%s" % (json.dumps(1000.0 * mean_time)))
        print("__results.throughput__=%s" % (json.dumps(int(mean_throughput))))
        print("__exp.model_title__=%s" % (json.dumps(model_title)))
        print("__results.time_data__=%s" % (json.dumps((1000.0*times).tolist())))
    else:
        print("__results.status__=%s" % (json.dumps("failure")))

