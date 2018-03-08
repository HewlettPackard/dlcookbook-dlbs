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
"""
from __future__ import absolute_import
from __future__ import print_function
import sys
import traceback
import argparse
import timeit
import json
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from pytorch_benchmarks.model_factory import ModelFactory
from pytorch_benchmarks.dataset_factory import DatasetFactory

def get_effective_batch_size(opts):
    """Returns effective batch size

    :param dict opts: A dictionary containing benchmark parameters. Must contain
                      `batch_size`, `device` and optionally `num_gpus`.
    :return: Effective batch size.
    :rtype: int
    """
    num_devices = 1 if opts['device'] == 'cpu' else opts['num_gpus']
    return opts['batch_size'] * num_devices


def benchmark(opts):
    """An entry point for inference/training benchmarks.

    :param dict opts: A dictionary of parameters.
    :rtype: tuple
    :return: A tuple of (model_name, list of batch times)
    """
    assert 'model' in opts, "Missing 'model' in options."
    assert 'phase' in opts, "Missing 'phase' in options."
    assert opts['phase'] in ['inference', 'training'],\
           "Invalid value for 'phase' (%s). Must be 'inference' or 'training'." % (opts['phase'])

    opts['batch_size'] = opts.get('batch_size', 16)
    opts['num_warmup_batches'] = opts.get('num_warmup_batches', 10)
    opts['num_batches'] = opts.get('num_batches', 10)
    opts['device'] = opts.get('device', 'gpu')
    opts['num_gpus'] = opts.get('num_gpus', 1)
    opts['dtype'] = opts.get('dtype', 'float32')
    opts['data_dir'] = opts.get('data_dir', '')
    #opts['enable_tensor_core'] = opts.get('enable_tensor_core', False)

    model = ModelFactory.get_model(opts)
    if opts['phase'] == 'inference':
        return benchmark_inference(model, opts)
    return benchmark_training(model, opts)


def benchmark_inference(model, opts):
    """Benchmarks inference phase.

    :param obj model: A model to benchmark
    :param dict opts: A dictionary of parameters.
    :rtype: tuple
    :return: A tuple of (model_name, list of batch times)
    """
    if opts['phase'] != 'inference':
        raise "Phase in benchmark_inference func is '%s'" % opts['phase']
    if opts['device'] == 'gpu' and opts['num_gpus'] != 1:
        raise "GPU inference can only be used with one GPU (num_gpus: %d)." % opts['num_gpus']

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
    model.evaluate()
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
    :rtype: tuple
    :return: A tuple of (model_name, list of batch times)
    """
    if opts['phase'] != 'training':
        raise "Phase in benchmark_training func is '%s'" % opts['phase']

    criterion = nn.CrossEntropyLoss()
    with_cuda = opts['num_gpus'] >= 1
    if with_cuda:
        cudnn.benchmark = opts['cudnn_benchmark']
        cudnn.fastest = opts['cudnn_fastest']
        # Data parallel schema over all avaialble GPUs - use CUDA_VISIBLE_DEVICES to
        # set GPUs to use.
        model = nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
    half_precision = opts['dtype'] == 'float16'
    if half_precision:
        model = model.half()
        criterion = criterion.half()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    data_loader = DatasetFactory.get_data_loader(model, opts, get_effective_batch_size(opts))

    is_warmup = opts['num_warmup_batches'] > 0
    done = opts['num_warmup_batches'] == 0
    num_iterations_done = 0
    model.train()
    batch_times = np.zeros(opts['num_batches'])
    while not done:
        for batch_data, batch_labels in data_loader:
            start_time = timeit.default_timer()
            if with_cuda:
                batch_data = batch_data.cuda(non_blocking=True)
                batch_labels = batch_labels.cuda(non_blocking=True)
            data_var = torch.autograd.Variable(batch_data)
            labels_var = torch.autograd.Variable(batch_labels)
            if half_precision:
                data_var = data_var.half()

            # compute output
            output = model(data_var)
            loss = criterion(output, labels_var)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end_time = timeit.default_timer()

            # Track progress
            num_iterations_done += 1
            if is_warmup and num_iterations_done >= opts['num_warmup_batches']:
                is_warmup = False
                num_iterations_done = 0
            else:
                if opts['num_batches'] != 0:
                    batch_times[num_iterations_done-1] = end_time - start_time
                if num_iterations_done >= opts['num_batches']:
                    done = True
                    break

    return (model.module.name, batch_times)


def main():
    """Main worker function."""
    def str2bool(val):
        """Converts 'val' to boolean value."""
        return val.lower() in ('true', 'on', 't', '1')

    print("[PyTorch benchmakrs] torch.version.__version__=%s" % torch.version.__version__)
    print("[PyTorch benchmakrs] torch.cuda.is_available()=%s" % torch.cuda.is_available())
    print("[PyTorch benchmakrs] torch.version.cuda=%s" % torch.version.cuda)
    print("[PyTorch benchmakrs] torch.backends.cudnn.version()=%s" % torch.backends.cudnn.version())

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
        help="Per device batch size"
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
        '--num_gpus', type=int, required=False, default=1,
        help="Number of gpus to use. Use CUDA_VISIBLE_DEVICES to select those devices."
    )
    parser.add_argument(
        '--device', type=str, required=False, default='cpu',
        help="Comptue device, 'cpu' or 'gpu'"
    )
    parser.add_argument(
        '--dtype', required=False, default='float', choices=['float', 'float32', 'float16'],
        help="Precision of data variables: float(same as float32), float32 or float16."
    )
    parser.add_argument(
        '--data_dir', type=str, required=False, default='',
        help="Path to a dataset. "
    )
    parser.add_argument(
        '--data_backend', type=str, required=False, default='caffe_lmdb',
        choices=['caffe_lmdb', 'image_folder'],
        help="In case if --data_dir is present, this argument defines type of dataset. "
    )

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
        '--data_shuffle', nargs='?', const=True, default=False, type=str2bool,
        help="Enable/disable shuffling for both real/synthetic datasets."
    )
    parser.add_argument(
        '--num_loader_threads', type=int, required=False, default=4,
        help="Number of dataset loader threads."
    )

    args = parser.parse_args()

    try:
        if args.dtype == 'float':
            args.dtype = 'float32'
        if args.dtype not in ['float32', 'float16']:
            msg = "PyTorch only supports float32 and float16 data types. But found '%s'"
            raise ValueError(msg % args.dtype)
        if args.num_gpus == 0 and args.dtype != 'float32':
            msg = "In CPU mode, dtype must be float32. Num gpus=%d, dtype=%s"
            raise ValueError(msg % (args.num_gpus, args.dtype))

        opts = vars(args)
        opts['phase'] = 'inference' if args.forward_only else 'training'
        model_title, times = benchmark(opts)
    except Exception, err:
        #TODO: this is not happenning, program terminates earlier.
        # For now, do not rely on __results.status__=...
        times = np.zeros(0)
        model_title = 'Unk'
        print ("Critical error while running benchmarks (%s). See stacktrace below." % (str(err)))
        traceback.print_exc(file=sys.stdout)

    if times.size > 0:
        mean_time = np.mean(times)                                   # seconds
        mean_throughput = get_effective_batch_size(opts) / mean_time # images / sec
        print("__results.time__=%s" % (json.dumps(1000.0 * mean_time)))
        print("__results.throughput__=%s" % (json.dumps(int(mean_throughput))))
        print("__exp.model_title__=%s" % (json.dumps(model_title)))
        print("__results.time_data__=%s" % (json.dumps((1000.0*times).tolist())))
    else:
        print("__results.status__=%s" % (json.dumps("failure")))


if __name__ == '__main__':
    main()
