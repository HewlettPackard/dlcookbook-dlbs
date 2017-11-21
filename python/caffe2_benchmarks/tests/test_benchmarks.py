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
"""Unit tests for testing caffe2_benchmarks.caffe2_benchmarks.py function."""
from __future__ import print_function
import os
import unittest
import itertools
# now we can import the lib module
from caffe2.python import workspace
from caffe2.python import model_helper
import numpy as np
from caffe2_benchmarks.benchmarks import benchmark_inference
from caffe2_benchmarks.benchmarks import benchmark_training

#@unittest.skipIf('DLBS_TESTS_RUN_CAFFE2' not in os.environ, "Skipping Caffe2 Benchmarks Test")
class TestCaffe2Benchmarks(unittest.TestCase):
    """Class tests Inference, GPU and CPU training"""

    def setUp(self):
        workspace.ResetWorkspace()
        self.models = [
            'deep_mnist', 'eng_acoustic_model', 'sensor_net', 'alexnet',
            'vgg11', 'vgg13', 'vgg16', 'vgg19',
            'googlenet',
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'resnet200', 'resnet269']
        self.batch_sizes = [1, 2, 4]
        self.num_warmup_iters = 1
        self.num_batches = 1
        self.num_gpus = 1
        self.devices = ['gpu', 'cpu']
        self.gpus = ['0']
        self.gpu_skip_models = ['resnet200', 'resnet269'] # May be too deep to fit in my GPU memory

    def test_inference(self):
        """caffe2_benchmarks ->  TestCaffe2Benchmarks::test_inference    [Caffe2 CPU/GPU inference.]"""
        print("Testing inference")
        for params in itertools.product(self.models, self.batch_sizes, self.devices):
            if params[0] in self.gpu_skip_models:
                continue
            model = model_helper.ModelHelper(name=params[0])
            name, times = benchmark_inference(
                model,
                {'model':params[0], 'phase':'inference', 'batch_size':params[1],
                 'num_batches':self.num_batches, 'num_warmup_batches':self.num_warmup_iters,
                 'num_gpus':self.num_gpus, 'device':params[2], 'dtype':'float',
                 'enable_tensor_core':False}
            )
            self.assertEqual(len(times), self.num_batches)
            print("model=%s, name=%s, batch=%d, device=%s, time=%f" %\
                  (params[0], name, params[1], params[2], 1000.0*np.mean(times)))
            workspace.ResetWorkspace()

    def test_training_cpu(self):
        """caffe2_benchmarks ->  TestCaffe2Benchmarks::test_training_cpu [Caffe2 CPU training.]"""
        print("Testing CPU training")
        for params in itertools.product(self.models, self.batch_sizes):
            model = model_helper.ModelHelper(name=params[0])
            name, times = benchmark_training(
                model,
                {'model':params[0], 'phase':'training', 'batch_size':params[1],
                 'num_batches':self.num_batches, 'num_warmup_batches':self.num_warmup_iters,
                 'num_gpus':0, 'device':'cpu', 'dtype':'float',
                 'enable_tensor_core':False}
            )
            self.assertEqual(len(times), self.num_batches)
            print("model=%s, name=%s, batch=%d, device=cpu, time=%f" %\
                  (params[0], name, params[1], 1000.0*np.mean(times)))
            workspace.ResetWorkspace()

    def test_training_gpu(self):
        """caffe2_benchmarks ->  TestCaffe2Benchmarks::test_training_gpu [Caffe2 GPU training.]"""
        print("Testing GPU training")
        for params in itertools.product(self.models, self.batch_sizes, self.gpus):
            if params[0] in self.gpu_skip_models:
                continue
            model = model_helper.ModelHelper(name=params[0])
            name, times = benchmark_training(
                model,
                {'model':params[0], 'phase':'training', 'batch_size':params[1],
                 'num_batches':self.num_batches, 'num_warmup_batches':self.num_warmup_iters,
                 'num_gpus':len(params[2].split()), 'device':'gpu', 'dtype':'float',
                 'enable_tensor_core':False}
            )
            self.assertEqual(len(times), self.num_batches)
            print("model=%s, name=%s, batch=%d, gpus=%s, time=%f" %\
                  (params[0], name, params[1], params[2], 1000.0*np.mean(times)))
            workspace.ResetWorkspace()

if __name__ == '__main__':
    unittest.main()
