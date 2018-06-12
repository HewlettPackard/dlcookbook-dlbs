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
"""Prints results of a series of benchmarks.

Usage:

 >>> python bench_stats.py [PARAMETERS]

Parameters:

* ``--log_dir`` Scan this folder for *.log files. Scan recursively if
  ``--recursive`` flag is provided.
* ``--recursive`` Scan ``--log-dir`` folder recursively for log files.

Example:
   Scan folder './bvlc_caffe' for log files recursively and print out stats to a console

   >>> python bench_stats.py --log-dir ./bvlc_caffe --recursive
"""
from __future__ import print_function
import argparse
import json
import dlbs.python_version   # pylint: disable=unused-import
from dlbs.utils import IOUtils
from dlbs.logparser import LogParser

class BenchStats(object):
    """Class that finds log files and computes simple statistics on experiments."""

    @staticmethod
    def compute(log_dir, recursive):
        """ Finds files and compute experiments' statistics.

        :param std log_dir: Directory to search files for.
        :param bool recursive: If True, directory will be searched recursively.
        :return: Dictionary with experiment statistics.
        """
        files = IOUtils.find_files(log_dir, "*.log", recursive)
        benchmarks, failed_benchmarks = LogParser.parse_log_files(files)
        def _get(d, key, val=''):
            return d[key] if key in d else val

        stats = {
            'num_log_files': len(files),
            'num_failed_exps': 0,
            'num_successful_exps': 0,
            'failed_exps': {},
            'node_ids': set(),
            'node_titles': set(),
            'gpu_titles': set()
        }
        for bench in benchmarks:
            time_val = str(bench['results.time']).strip() if 'results.time' in bench else ''
            if not time_val:
                stats['num_failed_exps'] += 1
                if 'exp.id' not in bench:
                    print("[ERROR] No exp.id found in benchmark (%s)" % str(bench))
                    continue
                stats['failed_exps'][bench['exp.id']] = {
                    'msg': 'No %s time found in log file.' % _get(bench, 'exp.phase', 'PHASE_UNKNOWN'),
                    'log_file': _get(bench, 'exp.log_file', 'LOG_FILE_UNKNOWN'),
                    'phase': _get(bench, 'exp.phase', 'PHASE_UNKNOWN'),
                    'framework_title': _get(bench, 'exp.framework_title', 'FRAMEWORK_TITLE_UNKNOWN')
                }
            else:
                stats['num_successful_exps'] += 1
            #
            for key in [('exp.node_id', 'node_ids'), ('exp.node_title', 'node_titles'), ('exp.gpu_title', 'gpu_titles')]:
                if key[0] in bench:
                    stats[key[1]].add(bench[key[0]])

        for key in ['node_ids', 'node_titles', 'gpu_titles']:
            stats[key] = list(stats[key])
        return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', '--log-dir', type=str, required=True, default=None,
                        help="Scan this folder for *.log files. "\
                             "Scan recursively if --recursive is set.")
    parser.add_argument('--recursive', required=False, default=False, action='store_true',
                        help='Scan --log_dir folder recursively for log files.')
    args = parser.parse_args()

    stats = BenchStats.compute(args.log_dir, args.recursive)
    print(json.dumps(stats, sort_keys=False, indent=2))
