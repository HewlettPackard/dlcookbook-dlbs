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
import os,sys
import unittest
# append parent directory to import path
import env #pylint: disable=W0611
from dlbs.utils import ConfigurationLoader
import json
from dlbs.builder import Builder
from dlbs.processor import Processor
from pprint import pprint
import argparse
from dlbs.utils import DictUtils
class TestVariableExpansion(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--user_config_file','-u', type=str, required=True, default=None,help="User configuration file.")
        parser.add_argument('--bench_root', '-b', type=str, required=True, default=None,help="Value of $BENCH_ROOT")
        parser.add_argument('--dlbs_root',  '-d', type=str, required=True, default=None,help="Value of $DLBS_ROOT")
        parser.add_argument('--extension',  '-e', type=str, action="append", required=False, default=None,help="extension specified from the command line")
        parser.add_argument('--parameter',  '-p', type=str, action="append", required=False, default=None,help="parameter specified from the command line")
        parser.add_argument('--variable',   '-v', type=str, action="append", required=False, default=None,help="variable specified from the command line")
        self.opts = vars(parser.parse_args())
        self.config_path = os.path.join(os.path.dirname(__file__), '..', 'configs')
        self.user_config_file=self.opts['user_config_file']
        os.environ['BENCH_ROOT']=self.opts['bench_root']
        os.environ['DLBS_ROOT']=self.opts['dlbs_root']

    def test(self):
        """dlbs  ->  TestVariableExpansion::test                       [Loading default configuration.]"""
        # Get the defaults
        print('opts')
        print(self.opts)
        print("Reading default configuration files.")
        files, config, param_info = ConfigurationLoader.load(self.config_path)
        print("Read default configuration files: {}".format(files))
        # Now the user config.
        print("Reading user configuration file {}".format(self.user_config_file))
        if self.user_config_file is not None:
            with open(self.user_config_file) as file_obj:
                user_config = json.load(file_obj)
            # Update parameter information from user configuration.
            ConfigurationLoader.update_param_info(param_info, user_config, is_user_config=True)
            # Update existing benchmark configuration.
            ConfigurationLoader.update(config, ConfigurationLoader.remove_info(user_config))
        print("Reading parameters and variables from the command line")
        params={}
        if 'parameter' in self.opts:
            DictUtils.add(params,self.opts['parameter'], pattern='(.+?(?=[=]))=(.+)', must_match=True)
        print("params: ",params)
        variables={}
        if 'variables' in self.opts:
            DictUtils.add(variables,self.opts['variables'], pattern='(.+?(?=[=]))=(.+)', must_match=True)
        print("variables: ",variables)
        print

        # Check method returns object of expected type
        self.assertIs(type(files), list)
        self.assertIs(type(config), dict)
        self.assertIs(type(param_info), dict)
        # Check we have parameters and extensions sections
        self.assertIn('parameters', config)
        self.assertIn('extensions', config)
        for param in config['parameters']:
            self.assertFalse(
                isinstance(config['parameters'][param], dict),
                "In configuration dictionary parameter value cannot be a dictionary"
            )
            self.assertIn(param, param_info, "Missing parameter in parameter info dictionary.")
        print("Check values in parameter info object are always dictionaries containing three mandatory fields.")
        for param in param_info:
            self.assertTrue(
                isinstance(param_info[param], dict),
                "In parameter info dictionary a value must be a ditionary."
            )
            for field in ('val', 'type', 'desc'):
                self.assertIn(field, param_info[param])
        print("\n")
        print("---------------------------------------------------------------------------------------------")
        print("Generating the Plan.")
        print("---------------------------------------------------------------------------------------------")
        plan=Builder.build(config,params,variables)
        Processor(param_info).compute_variables(plan)
        pprint(plan)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    result = unittest.TestResult()
    suite.addTest(unittest.makeSuite(TestVariableExpansion))
    runner = unittest.TextTestRunner()
    print(runner.run(suite))
    print('result: ',result)
