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

import os
import unittest
# append parent directory to import path
import dlbs.tests.env  # pylint: disable=W0611
from dlbs.tests.test_config_base import ConfigTester


class TestConfigCaffe(ConfigTester):
    def __init__(self, *args, **kwargs):
        ConfigTester.__init__(self, *args, **kwargs)

    def setUp(self):
        self.setUpBase(files=['base.json', 'caffe.json'])

    def check_parameters(self, fork, fork_title, docker_image):
        self.build_plan({"exp.framework": fork + "_caffe", "DLBS_ROOT": ""})
        self.compute_vars(
            [],
            [("exp.framework_title", fork_title + " Caffe"), ("exp.framework_family", "caffe"),
             ("caffe.fork", fork), ('exp.docker_image', docker_image),
             ('caffe.docker_image', docker_image), (fork + '_caffe.docker_image', docker_image),
             ('runtime.EXPORT_CUDA_CACHE_PATH', 'CUDA_CACHE_PATH=/workspace/cuda_cache'),
             ('caffe.env', 'CUDA_CACHE_PATH=/workspace/cuda_cache')]
        )

    def check_bare_metal_parameters(self, fork):
        self.build_plan({
            "exp.framework": fork + "_caffe", "DLBS_ROOT": "", "exp.docker": False
        })
        expexted_env_vars = [
            "PATH=%s/projects/%s_caffe/build/tools:\$PATH",
            "LD_LIBRARY_PATH=:\$LD_LIBRARY_PATH",
            "CUDA_CACHE_PATH=/dev/shm/dlbs"
        ]
        self.compute_vars(
            [],
            [('runtime.EXPORT_CUDA_CACHE_PATH', 'CUDA_CACHE_PATH=/dev/shm/dlbs'),
             ('caffe.env', " ".join(expexted_env_vars) % (os.environ['HOME'], fork))]
        )

    def test_bvlc_01(self):
        self.check_parameters('bvlc', 'BVLC', 'hpe/bvlc_caffe:cuda9-cudnn7')

    def test_nvidia_01(self):
        self.check_parameters('nvidia', 'NVIDIA', 'hpe/nvidia_caffe:cuda9-cudnn7')

    def test_intel_01(self):
        self.check_parameters('intel', 'Intel', 'hpe/intel_caffe:cpu')

    def test_bare_metal(self):
        for fork in ('bvlc', 'nvidia', 'intel'):
            self.check_bare_metal_parameters(fork)

    def test_nvidia_precision_01(self):
        self.build_plan({'exp.framework': 'nvidia_caffe', 'DLBS_ROOT': ''})
        for dtype in ('float32', 'float16'):
            nvp = 'FLOAT' if dtype == 'float32' else 'FLOAT16'
            self.compute_vars(
                [('exp.dtype', dtype)],
                [('nvidia_caffe.precision', dtype), ('nvidia_caffe.solver_precision', nvp),
                 ('nvidia_caffe.forward_precision', nvp), ('nvidia_caffe.backward_precision', nvp),
                 ('nvidia_caffe.forward_math_precision', nvp), ('nvidia_caffe.backward_math_precision', nvp)]
            )

    def test_nvidia_precision_02(self):
        self.build_plan({'exp.framework': 'nvidia_caffe', 'DLBS_ROOT': ''})
        ground_truth = {
            'float32': ['FLOAT', 'FLOAT', 'FLOAT'],
            'float16': ['FLOAT16', 'FLOAT16', 'FLOAT16'],
            'mixed': ['FLOAT', 'FLOAT16', 'FLOAT']
        }
        for nvp in ('float32', 'float16', 'mixed'):
            self.compute_vars(
                [('nvidia_caffe.precision', nvp)],
                [('nvidia_caffe.solver_precision', ground_truth[nvp][0]),
                 ('nvidia_caffe.forward_precision', ground_truth[nvp][1]),
                 ('nvidia_caffe.backward_precision', ground_truth[nvp][1]),
                 ('nvidia_caffe.forward_math_precision', ground_truth[nvp][2]),
                 ('nvidia_caffe.backward_math_precision', ground_truth[nvp][2])]
            )


if __name__ == '__main__':
    unittest.main()
