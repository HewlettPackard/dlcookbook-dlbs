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
""":py:class:`dlbs.logparser.LogParser` class parses log files and extracts experiments' parameters.

It can parse any other files that contain key-value items according to a specific
format. See :py:meth:`~dlbs.logparser.LogParser.parse_log_file` for details.

* Named arguments

  * ``--summary-file FILE_NAME``  Write summary of experiments into this JSON file.
  * ``--log-dir LOG_DIR``         Scan this folder for *.log files. Use ``--recursive`` flag to\
                                  search log files in subdirectories. Ignored if log files are\
                                  present.
  * ``--recursive``               Scan ``--log-dir`` folder recursively for log files.
  * ``--keys KEY1 KEY2 ...``      Parameters to extract from log files. If not set or empty,\
                                  all parameters are returned.
  * ``--strict``                  If set, include in the summary only those experiments that \
                                  contain all keys specified with ``--keys`` arg.
* Positional arguments

  * ``FILE1 FILE2 ...``           Log files to parse. If set, ``--log_dir`` parameter is ignored.

Example 1
    Parse one file and print results to a standard output. It should print out
    a whole bunch of parameters.

    >>> python logparser.py ./bvlc_caffe/alexnet_2.log

Example 2
    If we are intrested only in some of the parameters, we can specify them on a
    command line with ``--keys`` command line argument. That's OK if some of these
    parameters are not the log files.

    >>> python logparser.py ./bvlc_caffe/alexnet_2.log --keys "exp.framework_title" "exp.model_title"  "exp.effective_batch" "results.training_time" "results.inference_time"


Example 3
    It's possible to specify as many log files as you want:

    >>> python logparser.py ./bvlc_caffe/*.log --keys "exp.framework_title" "exp.model_title"  "exp.effective_batch" "results.training_time" "results.inference_time"

Example 4
    It's also possible to specify directory. In case of directory, a switch
    ``--recursive`` can be used to find log files in that directory and all its
    subdirectories

    >>> python logparser.py --log-dir ./bvlc_caffe --recursive --keys "exp.framework_title" "exp.model_title"  "exp.effective_batch" "results.training_time" "results.inference_time"

Example 5
    Finally, the summary can be written to a file

    >>> python logparser.py --summary-file ./bvlc_caffe/summary.json --log-dir ./bvlc_caffe --recursive \\
    >>>                     --keys "exp.gpus" "exp.framework_title" "exp.model_title"  "exp.effective_batch" "results.training_time" "results.inference_time" "exp.framework_id"
"""
from __future__ import print_function
import os
import sys
import json
import argparse
import gzip
import math
import dlbs.python_version   # pylint: disable=unused-import
from dlbs.utils import DictUtils
from dlbs.utils import IOUtils
from dlbs.processor import Processor

class LogParser(object):
    """Parser for log files produced by Deep Learning Benchmarking Suite."""

    @staticmethod
    def parse_log_files(filenames, opts=None):
        """ Parses files and returns their parameters.

        :param list filenames: List of file names to parse.
        :param dict opts:      Dictionary of options.

        :rtype:  tuple<list, list>
        :return: A tuple of two lists - succeeded and failed benchmarks
        """
        opts = {} if opts is None else opts
        for key in ('filter_params', 'filter_query', 'output_params'):
            DictUtils.ensure_exists(opts, key)
        DictUtils.ensure_exists(opts, 'failed_benchmarks', 'discard')
        DictUtils.ensure_exists(opts, '_extended_params', {})

        succeeded_benchmarks = []
        failed_benchmarks = []
        for filename in filenames:
            # Parse log file
            params = LogParser.parse_log_file(filename)
            # Check if this benchmark does not match filter
            if len(params) == 0 or \
               not DictUtils.contains(params, opts['filter_params']) or \
               not DictUtils.match(params, opts['filter_query']):
                continue
            # Add extended parameters and compute them
            if len(opts['_extended_params']) > 0:
                params.update(opts['_extended_params'])
                Processor().compute_variables([params])
                #params = params[0]
            # Identify is this benchmark succeeded of failed.
            succeeded = 'results.throughput' in params and \
                        isinstance(params['results.throughput'], (int, long, float)) and \
                        params['results.throughput'] > 0
            # Get only those key/values that need to be serialized
            params = DictUtils.subdict(params, opts['output_params'])
            # Append benchmark either to succeeded or failed list
            if succeeded:
                succeeded_benchmarks.append(params)
            else:
                if opts['failed_benchmarks'] == 'keep':
                    succeeded_benchmarks.append(params)
                elif opts['failed_benchmarks'] == 'keep_separately':
                    failed_benchmarks.append(params)
            #
        return (succeeded_benchmarks, failed_benchmarks)

    @staticmethod
    def parse_log_file(filename):
        """ Parses one log file.

        Parameters are defined in that file as key-value pairs. Values must be
        json parsable strings. Every key has a prefix and a suffix equal to ``__``
        (two underscores), for instance:

        * __exp.device_batch__= 16
        * __results.training_time__= 33.343

        Parameters are keys without prefixes and suffixes i.e. 'exp.device_batch'
        and 'results.training_time' are parameter names from above example.

        :param str filename: Name of a file to parse.
        :return: Dictionary with experiment parameters.
        :rtype: dict
        """
        #       __(.+?(?=__[=]))__=(.+)
        # [ \t]*__(.+?(?=__[ \t]*[=]))__[ \t]*=(.+)
        exp_params = {}
        with open(filename) as logfile:
            # The 'must_match' must be set to false. It says that not every line
            # in a log file must match key-value pattern.
            DictUtils.add(
                exp_params,
                logfile,
                pattern='[ \t]*__(.+?(?=__[ \t]*[=]))__[ \t]*=(.+)',
                must_match=False
            )
        return exp_params


def parse_args():
    """ Parse command line arguments.

    :rtype: dict
    :return: Dictionary of command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'inputs', nargs='*', default=None,
        help="Log files / directories to parse. It is a list of file names "\
             "and/or directories to search for log files."
    )
    parser.add_argument(
        '--recursive', required=False, default=False, action='store_true',
        help="Scan --log-dir folder recursively for log files."
    )
    parser.add_argument(
        '--output_file', '--output-file', type=str, required=False, default=None,
        help="Write summary of experiments into this file. Two types of files "\
             "are supported: *.json and *.json.gz. If multiple output files are "\
             "requested, the actual name will be '*_INDEX.json' or *_INDEX.json.gz. "\
             "If user requests to keep failed benchmarks separately, the name of "\
             "that file will be *_failed.json or *_failed.json.gz."
    )
    parser.add_argument(
        '--output_params', '--output-params', type=str, required=False, default=None,
        help="Parameters that will go into output files. If not specified, all"\
             "parameters are serialized. Keep in mind that will all parameters"\
             "output files may become qute large - you may want to use *.json.gz"\
             "pattern for output file to write gzipped files."
    )
    # Only one of benchmarks-per-file or num-output-files can present.
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--benchmarks_per_file', '--benchmarks-per-file', type=int, required=False, default=None,
        help="Maximal number of benchmarks per one output file. By default, "\
             "all benchmarks go into one file. Is not compatible with "\
             "--num_output_files argument."
    )
    group.add_argument(
        '--num_output_files', '--num-output-files', type=int, required=False, default=None,
        help="Number of output files. Algorithm is naive and may result in "\
             "output files being of different size: determine total number of"\
             "log files (N) and write first N/num-output-files benchmarks into"\
             "first file etc. Is not compatible with --benchmarks_per_file "\
             "argument."
    )
    #
    parser.add_argument(
        '--failed_benchmarks', '--failed-benchmarks', type=str, required=False, default='discard',
        choices=set(("keep", "discard", "keep_separately")),
        help="Action for failed benchmarks - those that do not contain positive "\
             "value for 'results.throughput' parameter."
    )
    # Only one filter specification is allowed
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--filter_params', '--filter-params', type=str, required=False, default=None,
        help="A comma separated list of parameters to use as a filter. Only "\
             "those benchmarks will be serialized that contain all parameters "\
             "defined here."
    )
    group.add_argument(
        '--filter_query', '--filter-query', type=str, required=False, default=None,
        help="A JSON dictionary that sets constraints parameter names. In order "\
             "to match, benchmark must contain all parameters defined in query "\
             "with exactly the same values. Format is similar to 'condition' "\
             "section in extensions."
    )
    #
    parser.add_argument(
        '-P', action='append', required=False, default=[],
        help="Parameters to add. Can be usefull to quickly add new parameters. "\
             "Must be valid json parsable dictionary. Values in this dictionary "\
             "can reference existing experiment parameters. If referenced parameter "\
             "does not exist, exception is thrown."
    )
    opts = vars(parser.parse_args())
    #
    output_file = opts['output_file']
    if output_file is not None:
        if os.path.exists(output_file):
            raise ValueError("File for benchmarks exists (%s)" % output_file)
        if output_file.endswith('.json.gz'):
            opts['_gz'] = True
            opts['_ext'] = 'json.gz'
            opts['_output_file_without_ext'] = output_file[:-8]
        elif output_file.endswith('.json'):
            opts['_gz'] = False
            opts['_ext'] = 'json'
            opts['_output_file_without_ext'] = output_file[:-5]
        else:
            raise ValueError("Output file must end with '.json.gz' or '.json'.")
    #
    if opts['failed_benchmarks'] == 'keep_separately':
        if output_file is None:
            raise ValueError("Cannot write failed benchmarks to separate file when main output file is none (console)")
        opts['_failed_file'] = "%s_failed.%s" % (opts['_output_file_without_ext'], opts['_ext'])
        if os.path.exists(opts['_failed_file']):
            raise ValueError("File for failed benchmarks exists (%s)" % opts['_failed_file'])
    if opts['output_params']:
        opts['output_params'] = opts['output_params'].strip(' \t,').split(',')
    if opts['filter_params']:
        opts['filter_params'] = opts['filter_params'].strip(' \t,').split(',')
    if opts['filter_query']:
        opts['filter_query'] = json.loads(opts['filter_query'])
    #
    extended_params = {}
    for ep in opts['P']:
        print(ep)
        extended_params.update(json.loads(ep))
    opts['_extended_params'] = extended_params
    #
    return opts


def main():
    """Does all log parsing work."""
    opts = parse_args()

    files = IOUtils.gather_files(opts['inputs'], "*.log", opts['recursive'])
    succeeded, failed = LogParser.parse_log_files(files, opts)

    def _dump_data(file_name, opts, data):
        with gzip.open(file_name, 'wb') if opts['_gz'] is True else open(file_name, 'w') as file_obj:
            json.dump({'data': data}, file_obj, indent=4)

    if opts['output_file'] is None:
        json.dump(succeeded, sys.stdout, indent=4, sort_keys=True)
        print ("")
    else:
        IOUtils.mkdirf(opts['output_file'])
        output_files = []
        if len(failed) > 0:
            _dump_data(opts['_failed_file'], opts, failed)
            output_files.append(opts['_failed_file'])

        num_benchmarks = len(succeeded)
        if opts['num_output_files'] is not None:
            opts['benchmarks_per_file'] = int(math.ceil(float(num_benchmarks) / opts['num_output_files']))

        if opts['benchmarks_per_file'] is not None:
            file_index = 0
            while True:
                start_index = file_index * opts['benchmarks_per_file']
                end_index = min(start_index + opts['benchmarks_per_file'], num_benchmarks)
                file_name = IOUtils.get_non_existing_file(
                    "%s_%d.%s" % (opts['_output_file_without_ext'], file_index, opts['_ext'])
                )
                _dump_data(
                    file_name,
                    opts,
                    succeeded[start_index:end_index]
                )
                output_files.append(file_name)
                if end_index >= num_benchmarks:
                    break
                file_index += 1
        else:
            _dump_data(opts['output_file'], opts, succeeded)
            output_files.append(opts['output_file'])
        print("Log parser summary.")
        print("Following files have been created:")
        json.dump(output_files, sys.stdout, indent=4, sort_keys=True)
        print ("")


if __name__ == "__main__":
    main()
