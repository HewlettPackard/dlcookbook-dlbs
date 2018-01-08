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
"""These unit tests test dlbs.utils.ConfigurationLoader class methods."""
import os
import unittest
# append parent directory to import path
import env #pylint: disable=W0611
from dlbs.utils import ConfigurationLoader

class TestConfigurationLoader(unittest.TestCase):

    params = {
        'exp': [
            "framework", "model", "env", "warmup_iters", "bench_iters", "phase",
            "device_batch", "gpus", "num_gpus", "device", "dtype", "enable_tensor_core",
            "simulation", "bench_root", "framework_id", "id", "effective_batch",
            "exp_path", "log_file", "force_rerun", "docker.launcher", "sys_info"
        ],
        'runtime': [
            "limit_resources", "cuda_cache", "bind_proc"
        ],
        'sys': [
            "plan_builder.var_order", "plan_builder.method"
        ],
        'resource_monitor': [
            "enabled", "pid_file_folder", "launcher", "data_file", "frequency"
        ],
        'tensorflow': [
            "launcher", "python_path", "env", "var_update", "use_nccl",
            "local_parameter_device", "data_dir", "data_name", "distortions",
            "num_intra_threads", "resize_method", "args", "docker.image",
            "docker.args", "host.libpath"
        ],
        'mxnet': [
            "launcher", "bench_path", "cudnn_autotune", "kv_store", "data_dir",
            "args", "host.python_path", "host.libpath"
        ],
        'caffe2': [
            "launcher", "data_dir", "data_backend", "dtype", "enable_tensor_core",
            "args", "docker.image", "docker.args", "bench_path", "host.python_path",
            "host.libpath"
        ],
        'caffe': [
            "launcher", "fork", "phase", "action", "model_file", "solver_file",
            "model_dir", "solver", "args", "data_dir", "mirror", "data_mean_file",
            "data_backend", "host.path", "docker.image", "docker.args",
        ],
        'bvlc_caffe': [
            "host.path", "host.libpath", "docker.image",
        ],
        'intel_caffe': [
            "host.path", "host.libpath", "docker.image",
        ],
        'nvidia_caffe': [
            "host.path", "host.libpath", "docker.image", "solver_precision",
            "forward_precision", "forward_math_precision",
            "backward_precision", "backward_math_precision"
        ],
        'tensorrt': [
            "launcher", "args", "model_file", "model_dir", "docker.image", 
            "docker.args", "profile", "input", "output", "host.path", "host.libpath"
        ]
    }

    def setUp(self):
        self.config_path = os.path.join(os.path.dirname(__file__), '..', 'configs')
        self.config_files = ['base.json', 'caffe.json', 'caffe2.json', 'mxnet.json',
                             'tensorflow.json', 'tensorrt.json']
        self.config_files.sort()

    def test(self):
        """dlbs  ->  TestConfigurationLoader::test                       [Loading default configuration.]"""
        files, config = ConfigurationLoader.load(self.config_path)
        self.assertIs(type(config), dict)
        #
        file_names = [os.path.basename(f) for f in files]
        file_names.sort()
        self.assertEqual(file_names, self.config_files)
        #
        self.assertIn('parameters', config)
        self.assertIn('extensions', config)
        #
        for ns in TestConfigurationLoader.params:
            for param in TestConfigurationLoader.params[ns]:
                self.assertIn(ns + '.' + param, config['parameters'])


if __name__ == '__main__':
    unittest.main()
