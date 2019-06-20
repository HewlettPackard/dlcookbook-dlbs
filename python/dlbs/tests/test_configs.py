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
import json
# append parent directory to import path
import dlbs.tests.env   # pylint: disable=W0611
import dlbs


class TestConfigs(unittest.TestCase):
    def setUp(self):
        pass

    def test_json_ok(self):
        """dlbs  ->  TestConfigs::test_json_ok                           [Configuration JSONs are ok.]"""
        configs_dir = os.path.join(os.path.dirname(dlbs.__file__), 'configs')
        config_files = [os.path.join(configs_dir, f) for f in os.listdir(configs_dir) if f.endswith('.json')]
        for config_file in config_files:
            with open(config_file) as f:
                try:
                    json.load(f)
                    # print("OK: " + config_file)
                except ValueError as error:
                    print("JSON Configuration file is invalid: %s" % config_file)
                    raise error


if __name__ == '__main__':
    unittest.main()
