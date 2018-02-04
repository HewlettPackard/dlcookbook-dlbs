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
import sys
import json
import argparse
import dlbs.python_version   # pylint: disable=unused-import
from dlbs.utils import DictUtils
from dlbs.utils import IOUtils

class LogParser(object):
    """Parser for log files produced by Deep Learning Benchnarking Suite."""

    @staticmethod
    def parse_log_files(filenames, keys=None):
        """ Parses files and returns their parameters.

        :param list filenames: List of file names to parse.
        :param list keys:      If not None, only these keys are added to parameters.

        :return: List of objects where every object is a dictionary containing\
                 parameters of experiment.
        :rtype: list
        """
        exps_params = []
        for filename in filenames:
            exps_params.append(LogParser.parse_log_file(filename, keys))
        return exps_params

    @staticmethod
    def parse_log_file(filename, keys=None):
        """ Parses one log file.

        Parameters are defined in that file as key-value pairs. Values must be
        json parsable strings. Every key has a prefix and a suffix equal to ``__``
        (two underscores), for instance:

        * __exp.device_batch__= 16
        * __results.training_time__= 33.343

        Parameters are keys without prefixes and suffixes i.e. 'exp.device_batch'
        and 'results.training_time' are parameter names from above example.

        If keys is not None, only those parameters are returned that in this list.

        :param str filename: Name of a file to parse.
        :param list keys:    If not None, only these keys are added to parameters else\
                             all found keys are added.
        :return: Dictionary with experiment parameters.
        :rtype: dict
        """
        #       __(.+?(?=__[=]))__=(.+)
        # [ \t]*__(.+?(?=__[ \t]*[=]))__[ \t]*=(.+)
        exp_params = {}
        with open(filename) as logfile:
            # The 'must_match' must be set to false. It says that not every line
            # in a log file must match key-value pattern.
            DictUtils.add(exp_params, logfile, pattern='[ \t]*__(.+?(?=__[ \t]*[=]))__[ \t]*=(.+)', must_match=False, add_only_keys=keys)
        return exp_params


def main():
    """Does all log parsing work."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary-file', type=str, required=False, default=None, help='Write summary of experiments into this JSON file.')
    parser.add_argument('--log-dir', type=str, required=False, default=None, help='Scan this folder for *.log files. Scan recursively if --recursive is set.')
    parser.add_argument('--recursive', required=False, default=False, action='store_true', help='Scan --log-dir folder recursively for log files.')
    parser.add_argument('--keys', nargs='*', required=False, help='Parameters to extract from log files. If not set or empty, all parameters are returned.')
    parser.add_argument('--strict', action='store_true', default=False, help='If set, serialzie only those results that contain all keys specified with --keys arg.')
    parser.add_argument('log_files', nargs='*', help='Log files to parse')
    args = parser.parse_args()

    files = []
    if len(args.log_files) > 0:
        files = args.log_files
    elif args.log_dir is not None:
        files = IOUtils.find_files(args.log_dir, "*.log", args.recursive)

    params = LogParser.parse_log_files(files, keys=args.keys)

    if args.strict and len(args.keys) > 0:
        filtered_params = []
        for param in params:
            param_ok = True
            for key in args.keys:
                if key not in param:
                    param_ok = False
                    #print ("skipping because missing field %s" % key)
                    break
            if param_ok:
                filtered_params.append(param)
        params = filtered_params

    summary = {"data": params}
    if args.summary_file is None:
        json.dump(summary, sys.stdout, indent=4, sort_keys=True)
        print ("")
    else:
        DictUtils.dump_json_to_file(summary, args.summary_file)


if __name__ == "__main__":
    main()
