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
"""Unit tests for testing mxnet_benchmarks.mxnet_benchmarks.py function."""
from __future__ import print_function
import os
import unittest
import itertools
import numpy as np
from mxnet_benchmarks.model_factory import ModelFactory
from mxnet_benchmarks.benchmarks import benchmark_inference
from mxnet_benchmarks.benchmarks import benchmark_training

#@unittest.skipIf('DLBS_TESTS_RUN_MXNET' not in os.environ, "Skipping MXNET Benchmarks Test")
class TestMXNetBenchmarks(unittest.TestCase):
    """Class tests Inference, GPU and CPU training"""

    def setUp(self):
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
        self.gpu_skip_models = ('resnet200', 'resnet269') # Too large for my dev GPU.

    def test_inference(self):
        """mxnet_benchmarks  ->  TestMXNetBenchmarks::test_inference     [MXNet CPU/GPU inference.]"""
        print("Testing inference")
        for params in itertools.product(self.models, self.batch_sizes, self.devices):
            model = ModelFactory.get_model({'model': params[0], 'phase': 'inference'})
            _, times = benchmark_inference(
                model,
                {'model':params[0], 'phase':'inference', 'batch_size':params[1],
                 'num_batches':self.num_batches, 'num_warmup_batches':self.num_warmup_iters,
                 'num_gpus':self.num_gpus, 'device':params[2]}
            )
            self.assertEqual(len(times), self.num_batches)
            for tm in times:
                self.assertGreater(tm, 0)
            print("model=%s, name=%s, batch=%d, device=%s, time=%f" %\
                  (params[0], model.name, params[1], params[2], 1000.0*np.mean(times)))
            del model

    def test_training_cpu(self):
        """mxnet_benchmarks  ->  TestMXNetBenchmarks::test_training_cpu  [MXNet CPU training.]"""
        print("Testing CPU training")
        for params in itertools.product(self.models, self.batch_sizes):
            model = ModelFactory.get_model({'model': params[0], 'phase': 'training'})
            _, times = benchmark_training(
                model,
                {'model':params[0], 'phase':'training', 'batch_size':params[1],
                 'num_batches':self.num_batches, 'num_warmup_batches':self.num_warmup_iters,
                 'num_gpus':0, 'device':'cpu', 'kv_store':'device'}
            )
            self.assertEqual(len(times), self.num_batches)
            for tm in times:
                self.assertGreater(tm, 0)
            print("model=%s, name=%s, batch=%d, device=cpu, time=%f" %\
                  (params[0], model.name, params[1], 1000.0*np.mean(times)))
            del model  # nope ...

    def test_training_gpu(self):
        """mxnet_benchmarks  ->  TestMXNetBenchmarks::test_training_gpu  [MXNet GPU training.]
        
        Now, this fails with OOM exception soon after start. How do I remove
        model and clear GPU memory?
        """
        print("Testing GPU training")
        for params in itertools.product(self.models, self.batch_sizes, self.gpus):
            if params[0] in self.gpu_skip_models:
                continue
            model = ModelFactory.get_model({'model': params[0], 'phase': 'training'})
            _, times = benchmark_training(
                model,
                {'model':params[0], 'phase':'training', 'batch_size':params[1],
                 'num_batches':self.num_batches, 'num_warmup_batches':self.num_warmup_iters,
                 'num_gpus':len(params[2].split()), 'device':'gpu', 'kv_store':'device'}
            )
            self.assertEqual(len(times), self.num_batches)
            for tm in times:
                self.assertGreater(tm, 0)
            print("model=%s, name=%s, batch=%d, gpus=%s, time=%f" %\
                  (params[0], model.name, params[1], params[2], 1000.0*np.mean(times)))
            del model  # nope ...


if __name__ == '__main__':
    unittest.main()
