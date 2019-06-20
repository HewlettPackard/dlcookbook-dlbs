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
"""
* Validate that every benchmark in ``input-file`` has mandatory parameters
  defined in ``params``

  $ python result_processor.py validate --input-file= --params=

* Filter benchmarks in ``input-file`` by throwing away those not containing
  specific parameters defined in ``params``. The filtered subset of benchmarks
  is written to ``output-file``.

  $ python result_processor.py filter --input-file= --params= --output-file=

* Update every benchmark in ``input-file`` by overriding values of specific
  parameters which value are defined in ``params``. The updated subset of
  benchmarks is written to ``output-file``.

  $ python result_processor.py update --input-file= --params= --output-file=

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import json
from collections import defaultdict
from dlbs.utils import Six
from dlbs.utils import DictUtils
from dlbs.processor import Processor


def load_json_file(file_name):
    """ Loads a json object from a file.

    :param str file_name: A file name to load JSON object from.
    :return: A loaded JSON object.
    """
    with open(file_name) as file_obj:
        return json.load(file_obj)


def get_params(params):
    """Loads parameters specified by params.

    :param str params: A JSON parsable string that defines how parameters
                       need to be loaded. See function comments on how it is
                       done.
    :return: A dictionary with keys being parameters and values being their
             values. Null value means no value - that's perfectly valid case.
    :rtype: dict

    The ``params`` is a JSON parsable string treated differently depending
    on its type:
    * ``string`` The value is a file name that contains JSON object
    * ``list``   The list of parameters
    * ``dict``   The dictionary that maps parameters to their values.

    If type is list or loaded JSON object is a list, it gets converted to
    dictionary with null values.
    """
    parsed_params = json.loads(params)
    if isinstance(parsed_params, Six.string_types):
        parsed_params = load_json_file(parsed_params)
    if isinstance(parsed_params, list):
        parsed_params = dict.fromkeys(parsed_params, None)
    if not isinstance(parsed_params, dict):
        raise ValueError("Invalid type of object that holds parameters (%s)" % type(parsed_params))
    return parsed_params


def validate_benchmarks(args):
    """Validates benchmarks ensuring every benchmark contains mandatory parameters.

    Also make sure `exp.id`s are unique.
    :param argparse args: Command line arguments.

    The following command line arguments are used:
    * ``args.input_file`` A file with benchmark results.
    * ``args.params``     Specification of mandatory parameters. For format,
                          read comments of ``get_params`` function
    """
    # Load benchmarks and parameters.
    benchmarks = load_json_file(args.input_file)['data']
    params = get_params(args.params)
    # Figure out missing parameters.
    missing_params = defaultdict(lambda: 0)
    exp_ids = set()            # All identifiers of experiments
    duplicates = False         # If two or more experiments have the same ID
    for benchmark in benchmarks:
        keys = [key for key in params if key not in benchmark]
        for key in keys:
            missing_params[key] += 1
        if 'exp.id' in benchmark:
            if benchmark['exp.id'] not in exp_ids:
                exp_ids.add(benchmark['exp.id'])
            else:
                duplicates = True
    # Report validation results.
    print("Number of benchmarks: %d" % len(benchmarks))
    if not missing_params and not duplicates:
        print("Benchmark validation result: SUCCESS")
    else:
        print("Benchmark validation result: FAILURE")
        if len(missing_params) > 0:
            print("missing parameters:")
            for missing_param in missing_params:
                print("\t%s: %d" % (missing_param, missing_params[missing_param]))
        if duplicates:
            print("Several benchmarks have same identifier (exp.id)")


def filter_benchmarks(args):
    """Filter benchmarks by removing those that do not contain provided parameters.

    :param argparse args: Command line arguments.

    The following command line arguments are used:
    * ``args.input_file`` A file with benchmark results.
    * ``args.params``     Specification of mandatory parameters. For format,
                          read comments of ``get_params`` function
    * ``args.output_file`` An output file with updated benchmark results.
    """
    # Load benchmarks and parameters
    input_benchmarks = load_json_file(args.input_file)['data']
    params = get_params(args.params)
    # Filter benchmarks
    output_benchmarks = []
    for input_benchmark in input_benchmarks:
        keep = True
        for key in params:
            if key not in input_benchmark or not input_benchmark[key]:
                keep = False
                break
        if keep:
            output_benchmarks.append(input_benchmark)
    # Report results and serialize
    print("Number of input benchmarks: %d" % len(input_benchmarks))
    print("Number of output benchmarks: %d" % len(output_benchmarks))
    DictUtils.dump_json_to_file({"data": output_benchmarks}, args.output_file)


def update_benchmarks(args):
    """Update benchmarks by overriding parameters provided by a user.

    :param argparse args: Command line arguments.

    The following command line arguments are used:
    * ``args.input_file`` A file with benchmark results.
    * ``args.params``     Specification of mandatory parameters. For format,
                          read comments of ``get_params`` function
    * ``args.output_file`` An output file with updated benchmark results.
    """
    # Load benchmarks and parameters.
    benchmarks = load_json_file(args.input_file)['data']
    prefix = '__'
    params = {prefix + k: v for k, v in get_params(args.params).items()}
    # Add prefixed parameters to all benchmarks.
    for benchmark in benchmarks:
        benchmark.update(params)
    # Process and compute variables
    Processor().compute_variables(benchmarks)
    # Replace prefix overwriting variables in case of a conflict
    prefixed_keys = params.keys()
    prefix_len = len(prefix)

    output_benchmarks = []
    for benchmark in benchmarks:
        for k in prefixed_keys:
            benchmark[k[prefix_len:]] = benchmark[k]
            del benchmark[k]
        if benchmark['exp.model'] != '':
            output_benchmarks.append(benchmark)
    benchmarks = output_benchmarks
    # Serialize updated benchmarks.
    DictUtils.dump_json_to_file({"data": benchmarks}, args.output_file)


def main():
    """Main function - parses command line args and processes benchmarks."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'action', type=str,
        help="Action to perform ('validate', 'filter', 'update')"
    )
    parser.add_argument(
        '--input_file', '--input-file', type=str, required=True, default=None,
        help='An input JSON file. This file is never modified.'
    )
    parser.add_argument(
        '--params', type=str, required=False, default=None,
        help="JSON array or object OR string. If string it's considered as a file name."
    )
    parser.add_argument(
        '--output_file', '--output-file', required=False, default=False,
        help="Output JSON file, possible, modified version of an input JSON file."
    )
    args = parser.parse_args()

    if args.action == 'validate':
        validate_benchmarks(args)
    elif args.action == 'filter':
        filter_benchmarks(args)
    elif args.action == 'update':
        update_benchmarks(args)
    else:
        raise ValueError("Action parameter has invalid value (%s). "
                         "Must be one of ['validate', 'filter', 'update']" % args.action)


if __name__ == '__main__':
    main()
