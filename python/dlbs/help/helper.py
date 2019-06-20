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
"""A class that manages all help-related activities.

It is responsible of printing help on parameters, performing full-text search
and suggesting default set of parameters for every framework. A sort of rule-based
knowledge approach

Show help on every parameter. Will print a lot of information.

$ python experimenter.py help --params

Show help on parameter based on regexp match

$ python experimenter.py help --params exp.device
$ python experimenter.py help --params exp.*

Perform full-text search in parameters description, case insensitive

$ python experimenter.py help --text cuda

Perform full-text search in a subset of parameters that match params

$ python experimenter.py help --params exp.device --text batch

Show most commonly used parameters for TensorFlow

$ experimenter.py help --frameworks tensorflow
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import os
import json
import argparse
import sys
from dlbs.utils import Six, ConfigurationLoader


class Helper(object):
    """The class that shows to a user various help messages on parameters,
    frameworks and most commonly used arguments
    """

    @staticmethod
    def main():
        """Entry point for running from a command line."""

        # If some of the parameters is not on a command line, its value is None.
        # If parameter is on a command line and value not there, list of that
        # parameter values is empty.
        parser = argparse.ArgumentParser()
        parser.add_argument('action', type=str, help="Action to perform. The only valid action is 'help'")
        parser.add_argument('--params', nargs='*', required=False,
                            help="Regular expression (s) that define which parameters to search for.")
        parser.add_argument('--text', nargs='*', required=False,
                            help="Regular expression (s) that define which parameter descriptions to search for. "
                                 "If --params is given, this is used to filter those results. If --params not given, "
                                 "search with this regexps in all parameters descriptions.")
        parser.add_argument('--frameworks', nargs='*', required=False,
                            help="Show help for these frameworks. It'll report the most commonly used parameters. "
                                 "If empty, list of supported framework is printed.")
        parser.add_argument("--no-colors", default=False, action="store_true",
                            help="If your terminal does not support colors, use this switch to disable colored output.")
        args = parser.parse_args()

        if args.params is None and args.text is None and args.frameworks is None:
            assert False, "No command line arguments provided. For help, use: python %s help --help" % (sys.argv[0])
        if args.params is not None or args.text is not None:
            if args.frameworks is not None:
                raise ValueError("The '--frameworks' argument must not be used with '--params' or '--text' arguments.")

        if args.frameworks is not None and len(args.frameworks) == 0:
            red = '\33[91m'   # Framework name Color
            grn = '\33[32m'   # Benchmark backend Color
            yellow = '\33[33m'   # Containers names

            def colr(txt, colr):
                return txt if args.no_colors else colr + txt + '\33[0m'

            help_msg = (
                colr("    TensorFlow\n", red) + 
                colr("        nvtfcnn", grn) + "           Highly optimized benchmark backend that provides best results on multi-GPU machines.\n"
                                "                          This backend must be used for best performance.\n" +
                                "                          NGC container: " + colr("nvcr.io/nvidia/tensorflow:18.07-py3\n", yellow) +
                colr("        nvcnn", grn) + "             Previous version of the 'nvtfcnn' benchmark backend.\n" +
                                "                          NGC container: " + colr("nvcr.io/nvidia/tensorflow:18.04-py3\n", yellow) +
                colr("        tensorflow", grn) + "        Google's TF_CNN_BENCHMARKS project. Will not provide very good performance with 8 GPUs.\n" +
                                "                          Reference DLBS container: " + colr("hpe/tensorflow:cuda9-cudnn7\n", yellow) +
                colr("    Caffe\n", red) + 
                colr("        bvlc_caffe", grn) + "        Benchmarks based on original BVLC Caffe implementation.\n" +
                                "                          Reference DLBS container: " + colr("hpe/bvlc_caffe:cuda9-cudnn7\n", yellow) +
                colr("        nvidia_caffe", grn) + "      Benchmarks based on NVIDIA's version of BVLC Caffe.\n" +
                                "                          NGC container: " + colr("nvcr.io/nvidia/caffe:18.05-py2\n", yellow) +
                colr("        intel_caffe", grn) + "       Intel's version of BVLC Caffe suitable for CPUs.\n" +
                                "                          Reference DLBS container: " + colr("hpe/intel_caffe:cuda9-cudnn7\n", yellow) +
                colr("    Caffes\n", red) + 
                colr("        caffe2", grn) +  "            Default benchmark backend for Caffe2 framework.\n" +
                                 "                          NGC container: " + colr("nvcr.io/nvidia/caffe2:18.05-py2\n", yellow) +
                colr("    MXNET\n", red) + 
                colr("        mxnet", grn) +  "             Default benchmark backend for MXNET framework.\n" +
                                 "                          NGC container: " + colr("nvcr.io/nvidia/mxnet:18.05-py2\n", yellow) +
                colr("    PyTorch\n", red) + 
                colr("        pytorch", grn) + "           Default benchmark backend for PyTorch framework.'\n" +
                                "                          NGC container: " + colr("nvcr.io/nvidia/pytorch:18.06-py3\n", yellow) +
                colr("    TensorRT\n", red) + 
                colr("        tensorrt", grn) +  "          Default benchmark backend for TensorRT inference engine.\n" +
                                 "                          DLBS container: " + colr("dlbs/tensorrt:18.10\n", yellow) +
                "\n"
                "Benchmark backends are specified with exp.framework parameter. For instance, to run benchmarks with fastest TensorFlow\n"
                "backend, provide the following:\n"
                "    CLI:  -Pexp.framework='\"nvtfcnn\"'\n"
                "    JSON: \"exp.framework\": \"nvtfcnn\"\n"
            )
            if not args.no_colors:
                print("\n(if your terminal does not support colors, rerun me with --no-colors switch)\n")
            print(help_msg)
            print ("Usage: python %s help --frameworks [FRAMEWORK [FRAMEWORK ...]]" % (sys.argv[0]))
            return

        helper = Helper()
        if args.frameworks is not None:
            framework_info = helper.help_with_frameworks(args.frameworks)
            for framework in framework_info:
                print("------------------------------------------------------")
                print("---------------------%s------------------------" % framework)
                print("------------------------------------------------------")
                Helper.print_param_help(framework_info[framework])
        else:
            params_info = helper.help_with_params(args.params, args.text)
            Helper.print_param_help(params_info)

    def __init__(self):
        """Initialize helper by parsing command line arguments.

        It's easier to do it manually. The format of command line is:
        experimenter.py action command command_parameters
        """
        _, _, self.param_info = ConfigurationLoader.load(
            os.path.join(os.path.dirname(__file__), '..', 'configs')
        )
        for key in self.param_info:
            pi = self.param_info[key]
            if 'desc' in pi and isinstance(pi['desc'], Six.string_types):
                pi['desc'] = [pi['desc']]
        with open(os.path.join(os.path.dirname(__file__), 'frameworks.json')) as file_obj:
            self.frameworks_help = json.load(file_obj)

    def help_with_params(self, param_patterns, text_patterns):
        """Search for help for specified parameters and/or optionally perform
        full text search in parameters descriptions.

        :param list param_patterns: List of regexps for parameter names.
        :param list text_patterns: List of regexps for parameter descriptions.
        :return: Dictionary that maps parameter name to its default value and
                 description message.

        If **param_patterns** is given and no **text_patterns** given, search for parameters.
        If **text_patterns** is given and no **param_patterns** given, do full text search in
        parameter descriptions.
        If **param_patterns** is given and **text_patterns** is given, search for parameters and
        filter matched parameters with regexps from **text_patterns**.

        So, we match **param_patterns** against parameter names and we match **text_patterns**
        against parameters' descriptions.
        """
        param_patterns = [] if param_patterns is None else param_patterns
        text_patterns = [] if text_patterns is None else text_patterns
        # Step 1: Match parameter names.
        if not param_patterns:
            # If parameters not given, pretend we have a complete list.
            matched_params = set(self.param_info.keys())
        else:
            matched_params = set()
            for param_pattern in param_patterns:
                regex = re.compile(param_pattern, re.I)
                for param in self.param_info.keys():
                    if regex.search(param):
                        matched_params.add(param)
        # Step 2: If text_patterns given, filter matched parameters
        if not text_patterns:
            filtered_params = matched_params.copy()
        else:
            filtered_params = set()
            for text_pattern in text_patterns:
                regex = re.compile(text_pattern, re.I)
                for matched_param in matched_params:
                    if matched_param not in filtered_params:
                        matched = False
                        for line in self.param_info[matched_param]['desc']:
                            if regex.search(line):
                                matched = True
                                break
                        if matched:
                            filtered_params.add(matched_param)
        # Step 3: Build response
        params_help = {}
        for filtered_param in filtered_params:
            params_help[filtered_param] = {
                'def_val': self.param_info[filtered_param]['val'],
                'help_msg': self.param_info[filtered_param]['desc']
            }
        return params_help

    def help_with_frameworks(self, frameworks):
        """Return most commonly used parameters for provided frameworks.

        :param list frameworks: List of frameworks identifiers.
        :return: Dictionary with parameters. Key is the framework identifier and
                 a value is a dictionary mapping parameter name to its default
                 value description message.
        """
        assert frameworks, "The frameworks list must not be empty"
        frameworks_help = {}

        def add_params(framework, params):
            """Add all params in **params** associated with **framework**."""
            for param in params:
                frameworks_help[framework][param] = {
                    'def_val': self.param_info[param]['val'],
                    'help_msg': self.param_info[param]['desc']
                }

        for framework in frameworks:
            frameworks_help[framework] = {}
            add_params(framework, self.frameworks_help['__base__'])
            if framework in ('bvlc_caffe', 'nvidia_caffe', 'intel_caffe'):
                add_params(framework, self.frameworks_help['caffe'])
            add_params(framework, self.frameworks_help[framework])
        return frameworks_help

    @staticmethod
    def print_param_help(params):
        """Prints help messages for found params.

        :param dict params: A dictionary that maps parameter name to its default
                            value and description message.
        """
        for param in sorted(params.keys()):
            print("%s = %s" % (param, params[param]['def_val']))
            for line in params[param]['help_msg']:
                print("\t" + line)

    @staticmethod
    def load_dicts(path):
        """Loads JSON files into one dictionary.

        :param str path: Path to search for JSON files
        :return: Dictionary
        """
        loaded_dict = {}
        dict_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('_help.json')]
        for dict_file in dict_files:
            with open(dict_file) as file_obj:
                loaded_dict.update(json.load(file_obj))
        return loaded_dict
