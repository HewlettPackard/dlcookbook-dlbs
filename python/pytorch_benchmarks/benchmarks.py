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
"""PyTorch benchmarks entry point module.
    A default implementation for PyTorch benchmarking environment.
    Based on this ImageNet example: https://github.com/pytorch/examples/blob/master/imagenet/main.py

    Version 18.10
        Code updates for multi-GPU benchmarks. Based on NVIDIA examples.
        New dependency - apex library (https://www.github.com/nvidia/apex)
"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import traceback
import argparse
import timeit
import json
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import network_to_half, prep_param_lists, model_grads_to_master_grads, master_params_to_model_params
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from pytorch_benchmarks.model_factory import ModelFactory
from pytorch_benchmarks.dataset_factory import DatasetFactory, DataPrefetcher

def get_effective_batch_size(opts):
    """Returns effective batch size

    :param dict opts: A dictionary containing benchmark parameters. Must contain
                      `batch_size`, `device` and optionally `num_gpus`.
    :return: Effective batch size.
    :rtype: int
    """
    return opts['batch_size'] * opts['world_size']


def benchmark(opts):
    """An entry point for inference/training benchmarks.

    :param dict opts: A dictionary of parameters.
    :rtype: tuple
    :return: A tuple of (model_name, list of batch times)
    """
    assert 'model' in opts, "Missing 'model' in options."
    assert 'phase' in opts, "Missing 'phase' in options."
    assert opts['phase'] in ['inference', 'training', 'data_ingestion'],\
           "Invalid value for 'phase' (%s). Must be 'inference', 'training' or 'data_ingestion'." % (opts['phase'])

    opts['batch_size'] = opts.get('batch_size', 16)
    opts['num_warmup_batches'] = opts.get('num_warmup_batches', 10)
    opts['num_batches'] = opts.get('num_batches', 10)
    opts['device'] = opts.get('device', 'gpu')
    opts['num_gpus'] = opts.get('num_gpus', 1)
    opts['dtype'] = opts.get('dtype', 'float32')
    opts['data_dir'] = opts.get('data_dir', '')
    #opts['enable_tensor_core'] = opts.get('enable_tensor_core', False)

    if opts['phase'] == 'data_ingestion':
        # A little bit ugly - models expect 'inference' or 'training'. Anyway, model will
        # not be used - we just need to know input shape and number of classes.
        model = ModelFactory.get_model(dict(opts, phase='inference'))
        return benchmark_data_ingestion(model, opts)

    model = ModelFactory.get_model(opts)
    opts['__input_shape'] = model.input_shape
    opts['__num_classes'] = model.num_classes
    opts['__name'] = model.name

    if opts['phase'] == 'inference':
        return benchmark_inference(model, opts)
    return benchmark_training(model, opts)


def benchmark_data_ingestion(model, opts):
    """ Benchmark data ingestion part of the workload.

    :param obj model: A neural network model. We need this model
                      to provide number of classes and input shapes
                      to data loader. The model itself will not be
                      instantiated.
    :param dict opts: A dictionary of parameters.
    :rtype: tuple
    :return: A tuple of (model_name, list of data ingestion times)
    """
    if opts['data_dir'] == '':
        raise ValueError('Data ingestion benchmarks: dataset not provided')
    data_loader = DatasetFactory.get_data_loader(opts, model.input_shape, model.num_classes)
    data_load_times = np.zeros(opts['num_batches'])
    num_iterations_done = 0
    is_warmup = (opts['num_warmup_batches'] > 0)
    done = (opts['num_batches'] == 0)
    end_time = timeit.default_timer()
    while not done:
        print ("[INFO] Starting new epoch")
        for _, _ in data_loader:
            data_load_time = timeit.default_timer() - end_time
            num_iterations_done += 1
            if is_warmup:
                if num_iterations_done >= opts['num_warmup_batches']:
                    is_warmup = False
                    num_iterations_done = 0
            else:
                data_load_times[num_iterations_done-1] = data_load_time
                if num_iterations_done >= opts['num_batches']:
                    done = True
                    break
            end_time = timeit.default_timer()
    return (model.name, data_load_times)


def benchmark_inference(model, opts):
    """Benchmarks inference phase.

    :param obj model: A model to benchmark
    :param dict opts: A dictionary of parameters.
    :rtype: tuple
    :return: A tuple of (model_name, list of batch times)
    """
    if opts['phase'] != 'inference':
        raise "Phase in benchmark_inference func is '%s'" % opts['phase']
    if opts['device'] == 'gpu' and opts['world_size'] != 1:
        raise "GPU inference can only be used with one GPU (world_size: %d)." % opts['world_size']

    # Batch, Channels, Height, Width
    data = autograd.Variable(torch.randn((opts['batch_size'],) + model.input_shape))
    if opts['device'] == 'gpu':
        # TODO: Is it good to enable cuDNN autotuning (batch size is fixed)?
        #   https://github.com/soumith/cudnn.torch#modes
        #   https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        # How many iterations do we need to get cuDNN decide what kernels to use?
        cudnn.benchmark = opts['cudnn_benchmark']
        cudnn.fastest = opts['cudnn_fastest']

        data = data.cuda()
        model = model.cuda()
    if opts['dtype'] == 'float16':
        data = data.half()
        model = model.half()
    model.eval()
    # Do warmup round
    for i in range(opts['num_warmup_batches']):
        model(data)
    # Do benchmark round
    batch_times = np.zeros(opts['num_batches'])
    for i in range(opts['num_batches']):
        start_time = timeit.default_timer()
        model(data)
        batch_times[i] = timeit.default_timer() - start_time
    return (model.name, batch_times)


def benchmark_training(model, opts):
    """Benchmarks training phase.

    :param obj model: A model to benchmark
    :param dict opts: A dictionary of parameters.
    :rtype: tuple:
    :return: A tuple of (model_name, list of batch times)
    """
    def _reduce_tensor(tensor):
        reduced = tensor.clone()
        dist.all_reduce(reduced, op=dist.reduce_op.SUM)
        reduced /= opts['world_size']
        return reduced

    if opts['phase'] != 'training':
        raise "Phase in benchmark_training func is '%s'" % opts['phase']

    opts['distributed'] = opts['world_size'] > 1
    opts['with_cuda'] = opts['device'] == 'gpu'
    opts['fp16'] = opts['dtype'] == 'float16'
    opts['loss_scale'] = 1

    if opts['fp16'] and not opts['with_cuda']:
        raise ValueError("Configuration error: FP16 can only be used with GPUs")

    if opts['with_cuda']:
        torch.cuda.set_device(opts['local_rank'])
        cudnn.benchmark = opts['cudnn_benchmark']
        cudnn.fastest = opts['cudnn_fastest']

    if opts['distributed']:
        dist.init_process_group(backend=opts['dist_backend'], init_method='env://')

    if opts['with_cuda']:
        model = model.cuda()
        if opts['dtype'] == 'float16':
            model = network_to_half(model)

    if opts['distributed']:
        model = DDP(model)

    if opts['fp16']:
        model_params, master_params = prep_param_lists(model)
    else:
        master_params = list(model.parameters())

    criterion = nn.CrossEntropyLoss()
    if opts['with_cuda']:
        criterion = criterion.cuda()
    optimizer = optim.SGD(master_params, lr=0.01, momentum=0.9, weight_decay=1e-4)

    data_loader = DatasetFactory.get_data_loader(opts, opts['__input_shape'], opts['__num_classes'])

    is_warmup = opts['num_warmup_batches'] > 0
    done = opts['num_warmup_batches'] == 0
    num_iterations_done = 0
    model.train()
    batch_times = np.zeros(opts['num_batches'])
    end_time = timeit.default_timer()
    while not done:
        prefetcher = DataPrefetcher(data_loader, opts)
        batch_data, batch_labels = prefetcher.next()
        while batch_data is not None:
            data_var = torch.autograd.Variable(batch_data)
            labels_var = torch.autograd.Variable(batch_labels)

            output = model(data_var)

            loss = criterion(output, labels_var)
            loss = loss * opts['loss_scale']
            # I'll need this for reporting
            #reduced_loss = _reduce_tensor(loss.data) if opts['distributed'] else loss.data

            if opts['fp16']:
                model.zero_grad()
                loss.backward()
                model_grads_to_master_grads(model_params, master_params)
                if opts['loss_scale'] != 1:
                    for param in master_params:
                        param.grad.data = param.grad.data / opts['loss_scale']
                optimizer.step()
                master_params_to_model_params(model_params, master_params)
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if opts['with_cuda']:
                torch.cuda.synchronize()

            # Track progress
            num_iterations_done += 1
            cur_time = timeit.default_timer()

            batch_data, batch_labels = prefetcher.next()

            if is_warmup:
                if num_iterations_done >= opts['num_warmup_batches']:
                    is_warmup = False
                    num_iterations_done = 0
            else:
                if opts['num_batches'] != 0:
                    batch_times[num_iterations_done-1] = cur_time - end_time
                if num_iterations_done >= opts['num_batches']:
                    done = True
                    break
            end_time = cur_time

    return (opts['__name'], batch_times)


def main():
    """Main worker function."""
    def str2bool(val):
        """Converts 'val' to boolean value."""
        return val.lower() in ('true', 'on', 't', '1')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, required=True, default='',
        help="A model to benchmark ('alexnet', 'googlenet' ...)"
    )
    parser.add_argument(
        '--forward_only', nargs='?', const=True, default=False, type=str2bool,
        help="Benchmark inference (if true) else benchmark training."
    )
    parser.add_argument(
        '--batch_size', type=int, required=True, default=None,
        help="Per device batch size. Effective batch will depend on number of GPUs/workers."
    )
    parser.add_argument(
        '--num_batches', type=int, required=False, default=100,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        '--num_warmup_batches', type=int, required=False, default=1,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        '--dtype', required=False, default='float', choices=['float', 'float32', 'float16'],
        help="Precision of data variables: float(same as float32), float32 or float16."
    )
    #
    parser.add_argument(
        '--data_dir', type=str, required=False, default='',
        help="Path to a dataset. "
    )
    parser.add_argument(
        '--data', type=str, required=False, default='synthetic',
        help="Data specifier. Ignored if --data_dir points to a real dataset. "\
             "If --data_dir is empty, synthetic data is used which is placed to "\
             "host memory. If --data is 'synthetic/device', synthetic data is "\
             "placed in device (GPU) memory."
    )
    parser.add_argument(
        '--data_backend', type=str, required=False, default='caffe_lmdb',
        choices=['caffe_lmdb', 'image_folder'],
        help="In case if --data_dir is present, this argument defines type of dataset. "
    )
    parser.add_argument(
        '--data_shuffle', nargs='?', const=True, default=False, type=str2bool,
        help="Enable/disable shuffling for both real/synthetic datasets."
    )
    parser.add_argument(
        '--data_loader_only', nargs='?', const=True, default=False, type=str2bool,
        help="Benchmark only data ingestion pipeline (data loader)."
    )
    #
    parser.add_argument(
        '--cudnn_benchmark', nargs='?', const=True, default=True, type=str2bool,
        help="Uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. "\
             "If this is set to false, uses some in-built heuristics that might not always "\
             "be fastest. By default cudnn_benchmark is set to TRUE. Setting to true will "\
             "improve performance, at the expense of using more memory. The input shape "\
             "should be the same for each batch, otherwise autotune will re-run for each "\
             "batch, causing a huge slow-down. More details are here: "\
             "https://github.com/soumith/cudnn.torch#modes"
    )
    parser.add_argument(
        '--cudnn_fastest', nargs='?', const=True, default=False, type=str2bool,
        help="Enables a fast mode for the Convolution modules - simply picks the fastest "\
             "convolution algorithm, rather than tuning for workspace size. By default, "\
             "cudnn.fastest is set to false. You should set to true if memory is not an "\
             "issue, and you want the fastest performance. More details are here: "\
             "https://github.com/soumith/cudnn.torch#modes"
    )
    parser.add_argument(
        '--num_loader_threads', type=int, required=False, default=4,
        help="Number of dataset loader threads."
    )
    # https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py
    parser.add_argument(
        '--dist_backend', default='nccl', type=str,
        help="Distributed backend. GPU training currently only achieves the best "\
             "performance using the NCCL distributed backend. Thus NCCL backend is "\
             "the recommended backend to use for GPU training."
    )
    parser.add_argument(
        '--local_rank', default=0, type=int,
        help="Rank of this process on a local node. If current compute device is a "\
        "GPU device, this process must use this GPU."
    )
    parser.add_argument(
        '--device', type=str, required=False, default='cpu',
        help="Comptue device, 'cpu' or 'gpu'"
    )

    args = parser.parse_args()

    opts = vars(args)
    opts['world_size'] = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opts['global_rank'] = int(os.environ['RANK']) if 'RANK' in os.environ else 0

    if opts['local_rank'] == 0:
        print("[PyTorch benchmakrs] torch.version.__version__=%s" % torch.version.__version__)
        print("[PyTorch benchmakrs] torch.cuda.is_available()=%s" % torch.cuda.is_available())
        print("[PyTorch benchmakrs] torch.version.cuda=%s" % torch.version.cuda)
        print("[PyTorch benchmakrs] torch.backends.cudnn.version()=%s" % torch.backends.cudnn.version())
        print("[PyTorch benchmarks] torch.distributed.is_available()=%s" % torch.distributed.is_available())
        ver = torch.version.__version__.split('.')
        if len(ver) >= 2 and ver[0] == '0' and int(ver[1]) < 4:
            print("[WARNING] Expecting PyTorch version 0.4 or above. Most likely something will fail.")

    try:
        if opts['dtype'] == 'float':
            opts['dtype'] = 'float32'
        if opts['dtype'] not in ['float32', 'float16']:
            msg = "PyTorch only supports float32 and float16 data types. But found '%s'"
            raise ValueError(msg % opts['dtype'])
        if opts['device'] == 'cpu' and opts['dtype'] != 'float32':
            msg = "In CPU mode, dtype must be float32. Device=%d, dtype=%s"
            raise ValueError(msg % (opts['device'], opts['dtype']))
        if opts['data_loader_only']:
            opts['phase'] = 'data_ingestion'
        else:
            opts['phase'] = 'inference' if args.forward_only else 'training'

        model_title, times = benchmark(opts)
    except Exception as err:
        #TODO: this is not happenning, program terminates earlier.
        # For now, do not rely on __results.status__=...
        times = np.zeros(0)
        model_title = 'Unk'
        print ("Critical error while running benchmarks (%s). See stacktrace below." % (str(err)))
        traceback.print_exc(file=sys.stdout)

    if opts['local_rank'] == 0:
        if times.size > 0:
            times = 1000.0 * times                                              # from seconds to milliseconds
            mean_time = np.mean(times)                                          # average time in milliseconds
            mean_throughput = get_effective_batch_size(opts) / (mean_time/1000) # images / sec
            print("__results.time__=%s" % (json.dumps(mean_time)))
            print("__results.time_std__=%s" % (json.dumps(np.std(times))))
            print("__results.throughput__=%s" % (json.dumps(int(mean_throughput))))
            print("__exp.model_title__=%s" % (json.dumps(model_title)))
            print("__results.time_data__=%s" % (json.dumps(times.tolist())))
        else:
            print("__results.status__=%s" % (json.dumps("failure")))


if __name__ == '__main__':
    main()
