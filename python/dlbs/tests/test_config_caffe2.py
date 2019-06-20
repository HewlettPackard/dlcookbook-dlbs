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
# append parent directory to import path
import dlbs.tests.env  # pylint: disable=W0611
from dlbs.tests.test_config_base import ConfigTester


class TestConfigCaffe2(ConfigTester):
    def __init__(self, *args, **kwargs):
        ConfigTester.__init__(self, *args, **kwargs)

    def setUp(self):
        self.setUpBase(files=['base.json', 'caffe2.json'])

    def check_parameters(self, docker_image):
        self.build_plan({"exp.framework": "caffe2", "DLBS_ROOT": ""})
        self.compute_vars(
            [],
            [("exp.framework_title", "Caffe2"), ("exp.framework_family", "caffe2"),
             ('exp.docker_image', docker_image), ('caffe2.docker_image', docker_image),
             ('runtime.EXPORT_CUDA_CACHE_PATH', 'CUDA_CACHE_PATH=/workspace/cuda_cache')]
        )

    def test_docker_images(self):
        self.check_parameters('nvcr.io/nvidia/caffe2:18.05-py2')


if __name__ == '__main__':
    unittest.main()
