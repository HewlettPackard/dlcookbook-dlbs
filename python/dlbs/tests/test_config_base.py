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
"""Unit tests to verify all json configs can be parsed."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import os
import copy
# append parent directory to import path
import dlbs.tests.env  # pylint: disable=W0611
from dlbs.utils import ConfigurationLoader
from dlbs.builder import Builder
from dlbs.processor import Processor


class ConfigTester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.plan = None
        self.config = None
        self.param_info = None

    def build_plan(self, params):
        self.plan = Builder.build(self.config, params=params, variables={})

    def setUpBase(self, files=None):
        if files is None:
            files = ['base.json']
        _, self.config, self.param_info = ConfigurationLoader.load(
            os.path.join(os.path.dirname(dlbs.__file__), 'configs'),
            files=files
        )
        self.plan = None

    def compute_vars(self, inputs, expected_outputs):
        plan = copy.deepcopy(self.plan)
        exp = plan[0]
        for input_param in inputs:
            exp[input_param[0]] = input_param[1]
        Processor(self.param_info).compute_variables(plan)
        for expected_output in expected_outputs:
            self.assertEqual(
                exp[expected_output[0]],
                expected_output[1],
                "Actual output %s = %s differs from expected %s." % (expected_output[0], str(exp[expected_output[0]]),
                                                                     expected_output[1])
            )


class TestConfigBase(ConfigTester):
    def __init__(self, *args, **kwargs):
        ConfigTester.__init__(self, *args, **kwargs)

    def setUp(self):
        self.setUpBase()
        # The 'exp.docker_image' variable's value depend on variable which name
        # depends on 'exp.framework' value. So, we need to disable it by setting
        # 'exp.docker_image' to an empty string.
        # Same applies for `exp.data_dir`.
        self.build_plan({
            "exp.framework": "tensorflow", "exp.framework_title": "TensorFlow",
            "exp.framework_ver": "1.4.0", "exp.docker_image": "", "DLBS_ROOT": "",
            "exp.proj": "TestConfigBase", "exp.model": "resnet50",
            "exp.model_title": "ResNet50", "exp.data_dir": ""
        })

    def test_base(self):
        self.assertEqual(len(self.plan), 1)
        self.compute_vars(
            [('exp.num_warmup_batches', 111), ('exp.num_batches', 777),
             ('exp.phase', 'inference'), ('exp.data', 'real'), ('exp.data_store', 'local-ssd'),
             ('exp.dtype', 'float16'), ('exp.use_tensor_core', True)],
            [('exp.status', 'ok'), ('exp.status_msg', ''), ('exp.proj', 'TestConfigBase'),
             ('exp.framework', 'tensorflow'), ('exp.framework_title', 'TensorFlow'),
             ('exp.framework_family', 'tensorflow'), ('exp.framework_ver', '1.4.0'),
             ('exp.framework_commit', ''), ('exp.model', 'resnet50'),
             ('exp.model_title', 'ResNet50'), ('exp.num_warmup_batches', 111),
             ('exp.num_batches', 777), ('exp.phase', 'inference'),
             ('exp.data', 'real'), ('exp.data_store', 'local-ssd'),
             ('exp.dtype', 'float16'), ('exp.use_tensor_core', True)]
        )

    def test_cpu(self):
        for num_nodes in range(1, 11):
            self.compute_vars(
                [('exp.gpus', ''), ('exp.num_nodes', num_nodes)],
                [('exp.replica_batch', 16), ('exp.effective_batch', 16*num_nodes),
                 ('exp.num_local_replicas', 1), ('exp.num_local_gpus', 0),
                 ('exp.num_replicas', num_nodes), ('exp.num_gpus', 0),
                 ('exp.device_type', 'cpu'), ('runtime.visible_gpus', ''),
                 ('runtime.EXPORT_CUDA_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES='),
                 ('exp.docker_launcher', 'docker')]
            )

    def test_gpu_01(self):
        for gpus in [('0', 1), ('0,1', 2), ('0,1,2,3', 4), ('0,1,2,3,4,5,6,7', 8)]:
            for num_nodes in range(1, 11):
                self.compute_vars(
                    [('exp.gpus', gpus[0]), ('exp.num_nodes', num_nodes)],
                    [('exp.replica_batch', 16), ('exp.effective_batch', 16*gpus[1]*num_nodes),
                     ('exp.num_local_replicas', gpus[1]), ('exp.num_local_gpus', gpus[1]),
                     ('exp.num_replicas', gpus[1]*num_nodes), ('exp.num_gpus', gpus[1]*num_nodes),
                     ('exp.device_type', 'gpu'), ('runtime.visible_gpus', gpus[0]),
                     ('runtime.EXPORT_CUDA_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES='+gpus[0]),
                     ('exp.docker_launcher', 'nvidia-docker')]
                )

    def test_gpu_02(self):
        for num_nodes in range(1, 11):
            self.compute_vars(
                [('exp.gpus', '0:1'), ('exp.num_nodes', num_nodes)],
                [('exp.replica_batch', 16), ('exp.effective_batch', 16*num_nodes),
                 ('exp.num_local_replicas', 1), ('exp.num_local_gpus', 2),
                 ('exp.num_replicas', num_nodes), ('exp.num_gpus', 2*num_nodes),
                 ('exp.device_type', 'gpu'), ('runtime.visible_gpus', '0,1'),
                 ('runtime.EXPORT_CUDA_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES=0,1'),
                 ('exp.docker_launcher', 'nvidia-docker')]
            )

    def test_gpu_03(self):
        for num_nodes in range(1, 11):
            self.compute_vars(
                [('exp.gpus', '0:1,2:3'), ('exp.num_nodes', num_nodes)],
                [('exp.replica_batch', 16), ('exp.effective_batch', 16*2*num_nodes),
                 ('exp.num_local_replicas', 2), ('exp.num_local_gpus', 4),
                 ('exp.num_replicas', 2*num_nodes), ('exp.num_gpus', 4*num_nodes),
                 ('exp.device_type', 'gpu'), ('runtime.visible_gpus', '0,1,2,3'),
                 ('runtime.EXPORT_CUDA_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES=0,1,2,3'),
                 ('exp.docker_launcher', 'nvidia-docker')]
            )


if __name__ == '__main__':
    unittest.main()
