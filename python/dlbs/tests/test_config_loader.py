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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest
# append parent directory to import path
import dlbs.tests.env  # pylint: disable=W0611
from dlbs.utils import ConfigurationLoader


class TestConfigurationLoader(unittest.TestCase):

    params = {
        'exp': [
            "status", "status_msg", "proj", "framework", "framework_title", 
            "framework_family", "framework_ver", "framework_commit", "model", 
            "model_title", "num_warmup_batches", "num_batches", "phase", "data", 
            "data_store", "dtype", "use_tensor_core", "effective_batch", 
            "replica_batch", "gpus", "num_local_replicas", "num_local_gpus",
            "num_replicas", "num_gpus", "device_type", "device_title", "cuda",
            "cudnn", "id", "log_file", "rerun", "node_id", "num_nodes", "node_nic",
            "docker", "docker_launcher", "docker_image", "docker_args"
        ],
        'runtime': [
            "launcher", "cuda_cache", "visible_gpus",
            "EXPORT_CUDA_VISIBLE_DEVICES", "EXPORT_CUDA_CACHE_PATH"
        ],
        'monitor': [
            "frequency", "pid_folder", "launcher"
        ],
        'tensorflow': [
            "launcher", "python_path", "env", "var_update", "use_nccl",
            "local_parameter_device", "data_dir", "data_name", "distortions",
            "num_intra_threads", "resize_method", "args", "docker_image",
            "docker_args", "host_libpath"
        ],
        'mxnet': [
            "launcher", "bench_path", "cudnn_autotune", "kv_store", "data_dir",
            "args", "host_python_path", "host_libpath"
        ],
        'caffe2': [
            "launcher", "data_dir", "data_backend", "args", "docker_image",
            "docker_args", "bench_path", "host_python_path", "host_libpath"
        ],
        'caffe': [
            "launcher", "env", "fork", "action", "model_file", "solver_file",
            "model_dir", "solver", "args", "data_dir", "mirror", "data_mean_file",
            "data_backend", "host_path", "docker_image", "docker_args",
        ],
        'bvlc_caffe': [
            "host_path", "host_libpath", "docker_image",
        ],
        'intel_caffe': [
            "host_path", "host_libpath", "docker_image",
        ],
        'nvidia_caffe': [
            "host_path", "host_libpath", "docker_image", "solver_precision",
            "forward_precision", "forward_math_precision",
            "backward_precision", "backward_math_precision",
            "precision"
        ],
        'tensorrt': [
            "launcher", "args", "model_file", "model_dir", "docker_image", 
            "docker_args", "profile", "input", "output", "host_path", "host_libpath"
        ],
        'pytorch': [
            "launcher", "env", "bench_path", "data_dir", "cudnn_benchmark",
            "cudnn_fastest", "data_shuffle", "num_loader_threads", "args",
            "docker_image", "docker_args", "host_python_path", "host_libpath",
            "data_backend"
        ]
    }

    def setUp(self):
        self.config_path = os.path.join(os.path.dirname(__file__), '..', 'configs')
        self.config_files = ['base.json', 'caffe.json', 'caffe2.json', 'mxnet.json',
                             'tensorflow.json', 'tensorrt.json', 'pytorch.json', 'nvcnn.json', 'nvtfcnn.json']
        self.config_files.sort()

    def test_update_param_info(self):
        pi = {}
        ConfigurationLoader.update_param_info(
            pi,
            {'parameters': {
                'p1': 1, 'p2': u'2', 'p3': '3', 'p4': ['1', '2', '3', '4'],
                'p5': False, 'p6': -3.33,
                'p7': {'val': '34', 'type': 'str', 'desc': 'Some desc'}
            }}
        )
        for p in ('p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'):
            self.assertIn(p, pi)
            for f in ('val', 'type', 'desc'):
                self.assertIn(f, pi[p])
        for s in (('p1', 'int', 1), ('p2', 'str', u'2'), ('p3', 'str', '3'),
                  ('p4', 'str', ['1', '2', '3', '4']), ('p5', 'bool', False),
                  ('p6', 'float', -3.33), ('p7', 'str', '34')):
            self.assertEqual(pi[s[0]]['type'], s[1])
            self.assertEqual(pi[s[0]]['val'],  s[2])

    def test_remove_info(self):
        config = ConfigurationLoader.remove_info(
            {'parameters': {
                'p1': 1, 'p2': u'2', 'p3': '3', 'p4': ['1', '2', '3', '4'],
                'p5': False, 'p6': -3.33,
                'p7': {'val': '34', 'type': 's3tr', 'desc': 'Some desc'}
            }}
        )
        for p in ('p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'):
            self.assertIn(p, config['parameters'])
        for s in (('p1', 1), ('p2', u'2'), ('p3', '3'),
                  ('p4', ['1', '2', '3', '4']), ('p5', False),
                  ('p6', -3.33), ('p7', '34')):
            self.assertEqual(config['parameters'][s[0]], s[1])

    def test(self):
        """dlbs  ->  TestConfigurationLoader::test                       [Loading default configuration.]"""
        files, config, param_info = ConfigurationLoader.load(self.config_path)
        # Check method returns object of expected type
        self.assertIs(type(files), list)
        self.assertIs(type(config), dict)
        self.assertIs(type(param_info), dict)
        # Check we load all configuration files
        file_names = [os.path.basename(f) for f in files]
        file_names.sort()
        self.assertEqual(file_names, self.config_files)
        # Check we have parameters and extensions sections
        self.assertIn('parameters', config)
        self.assertIn('extensions', config)
        # Check presence of standard parameters
        for ns in TestConfigurationLoader.params:
            for param in TestConfigurationLoader.params[ns]:
                self.assertIn(ns + '.' + param, config['parameters'])
        # Check that values in configuration are not dictionaries and always have
        # a parameter info object for every parameter
        for param in config['parameters']:
            self.assertFalse(
                isinstance(config['parameters'][param], dict),
                "In configuration dictionary parameter value cannot be a ditionary"
            )
            self.assertIn(param, param_info, "Missing parameter in parameter info dictionary.")
        # Check values in paramter info object are always dictionaries containing
        # three mandatory fields.
        for param in param_info:
            self.assertTrue(
                isinstance(param_info[param], dict),
                "In parameter info dictionary a value must be a ditionary."
            )
            for field in ('val', 'type', 'desc'):
                self.assertIn(field, param_info[param])

    def test_path_none(self):
        # None path must throw error
        with self.assertRaises(ValueError):
            ConfigurationLoader.load(None)
        # Non existing directory must trigger Value Error
        with self.assertRaises(ValueError):
            ConfigurationLoader.load('/dr3/f2t23f/tfwegh5/sgh3gw4/hh/')
        # Existing directory and non existing file must trigger ValueError
        with self.assertRaises(ValueError):
            ConfigurationLoader.load('/', files=['sfasdf23r23r23r2r23r.json'])


if __name__ == '__main__':
    unittest.main()
