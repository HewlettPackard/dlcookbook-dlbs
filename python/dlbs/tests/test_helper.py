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
"""Tests Helper class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
# append parent directory to import path
import dlbs.tests.env  # pylint: disable=W0611
from dlbs.help.helper import Helper
from dlbs.tests.test_config_loader import TestConfigurationLoader


class TestHelper(unittest.TestCase):
    def setUp(self):
        self.helper = Helper()

    def test_help_with_params_01(self):
        """dlbs  ->  TestHelper::test_help_with_params_01                [Tests for parameter helper #1]"""
        self.assertEqual(len(self.helper.help_with_params(['exp.status'], None)), 2)
        self.assertEqual(len(self.helper.help_with_params(['^exp.status$'], None)), 1)
        self.assertEqual(len(self.helper.help_with_params(['^exp.status$', '^exp.num_gpus$'], None)), 2)
        # These may not be required in general.
        for ns in TestConfigurationLoader.params:
            for param in TestConfigurationLoader.params[ns]:
                param_name = ns + '.' + param
                self.assertEqual(
                    len(self.helper.help_with_params(['^' + param_name + '$'], None)),
                    1,
                    "Missing help message for parameter '%s'" % param_name
                )

    def test_help_with_params_02(self):
        """dlbs  ->  TestHelper::test_help_with_params_02                [Tests for parameter helper #2]"""
        self.assertEqual(len(self.helper.help_with_params(None, ['cuda'])), 4)
        self.assertEqual(len(self.helper.help_with_params(None, ['cudnn'])), 4)
        self.assertEqual(len(self.helper.help_with_params(None, ['docker'])), 35)

    def test_help_with_params_03(self):
        """dlbs  ->  TestHelper::test_help_with_params_03                [Tests for parameter helper #3]"""
        self.assertEqual(len(self.helper.help_with_params(['docker'], ['TensorRT'])), 1)

    def test_frameworks_help_01(self):
        """dlbs  ->  TestHelper::test_frameworks_help_01                 [Tests for framework helper #1]"""
        for framework in ('tensorflow', 'caffe2', 'bvlc_caffe', 'intel_caffe', 'nvidia_caffe', 'mxnet', 'tensorrt'):
            self.assertEqual(len(self.helper.help_with_frameworks([framework])), 1)

    def test_frameworks_help_02(self):
        """dlbs  ->  TestHelper::test_frameworks_help_02                 [Tests for framework helper #2]"""
        for framework in self.helper.frameworks_help:
            for param in self.helper.frameworks_help[framework]:
                self.assertIn(param, self.helper.param_info)


if __name__ == '__main__':
    unittest.main()
