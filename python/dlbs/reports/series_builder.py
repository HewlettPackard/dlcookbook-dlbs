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
Parses log files or loads JSON file with parsed results and produces JSON
with series information that can be used to plot charts.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import json
import logging
from collections import defaultdict
from dlbs.utils import IOUtils
from dlbs.utils import DictUtils
from dlbs.logparser import LogParser
from dlbs.utils import Modules
if Modules.HAVE_NUMPY:
    import numpy as np
if Modules.HAVE_MATPLOTLIB:
    # Set matplotlib's backend to Agg. All this script needs to be able to
    # do is to write png files including hosts not running graphical server.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


class SeriesBuilder(object):
    """Creates a JSON object that can be used to plot charts, plots chart."""

    @staticmethod
    def build(benchmarks, args):
        """Creates a JSON object that can be used to plot charts.

        :param list benchmarks: An array of benchmarks
        :param obj args: A result of argparse.parse. Contains parameters defining
                         the chart.
        """
        series_filters = json.loads(args.series)
        # During pre-processing step, we store series as dictionaries mapping
        # X to Y. Then, we convert it into array.
        chart_data = {
            'ylabel': args.yparam,   # Benchmark parameter for Y-axis
            'xlabel': args.xparam,   # Benchmark parameter for X-axis
            'series': [],            # List of {'filters': dict(), 'data': dict()}
            'xvals': set()           # Possible values for X-axis
        }
        for series_filter in series_filters:
            chart_data['series'].append({'filters': series_filter, 'data': defaultdict(list)})
        # Iterate over each benchmark and see if it needs to go into series
        for benchmark in benchmarks:
            # Without 'x' or 'y' data we cannot do anything.
            if args.xparam not in benchmark or args.yparam not in benchmark:
                continue
            # Iterate over series (their filters)
            for idx, series_filter in enumerate(series_filters):
                # If we cannot match all keys from query, ignore it
                if not DictUtils.match(benchmark, series_filter, policy='strict'):
                    continue
                xval = str(benchmark[args.xparam])
                yval = benchmark[args.yparam]
                chart_data['series'][idx]['data'][xval].append(yval)
                chart_data['xvals'].add(xval)
        # Perform final aggregation
        reducers = {'min': min, 'max': max, 'avg': lambda arr: float(sum(arr)) / len(arr)}
        reducer = reducers[args.aggregation]
        baseline_xvalue_exists = True
        for series in chart_data['series']:
            # Reduce multiple matches
            for xval in series['data']:
                series['data'][xval] = reducer(series['data'][xval])
            # Check if normalization to a baseline X value is possible
            if args.baseline_xvalue and args.baseline_xvalue not in series['data']:
                baseline_xvalue_exists = False
        # In-series normalization with respect to baseline value. It's performed
        # only when all series can be normalized.
        if args.baseline_xvalue and baseline_xvalue_exists:
            for series in chart_data['series']:
                baseline_val = series['data'][args.baseline_xvalue]
                for xval in series['data']:
                    series['data'][xval] /= baseline_val
        # Normalization with respect to baseline series
        if args.baseline_series:
            # We will normalize only when all values from other series can be scaled
            # i.e. baseline series must contain values for x points found in all other
            # series
            baseline_series_norm_ok = True
            baseline_series = chart_data['series'][args.baseline_series]['data'].copy()
            for idx, series in enumerate(chart_data['series']):
                if idx == args.baseline_series:
                    continue
                if not baseline_series_norm_ok:
                    break
                for xval in series['data']:
                    if xval not in baseline_series:
                        baseline_series_norm_ok = False
                        break
            if baseline_series_norm_ok:
                for series in chart_data['series']:
                    for xval in series['data']:
                        series['data'][xval] = series['data'][xval] / baseline_series[xval]

        # Return series info
        chart_data['xvals'] = list(chart_data['xvals'])
        print(json.dumps(chart_data, indent=4))
        return chart_data

    @staticmethod
    def plot(chart_data, args):
        """Serializes series as graphical charts into a file.

        :param dict chart_data: A data for chart, result of 'build' static method.
        :param obj args: A result of argparse.parse. Contains parameters defining
                         the chart.
        """
        if not Modules.HAVE_NUMPY or not Modules.HAVE_MATPLOTLIB:
            msg = "This script needs Numpy (available=%s) and Matplotlib (available=%s)"
            print (msg % (Modules.HAVE_NUMPY, Modules.HAVE_MATPLOTLIB))
            return
        # Default chart options.
        chart_opts = {
            'title': 'Title',
            'xlabel': chart_data['xlabel'], 'ylabel': chart_data['ylabel'],
            'legend': [str(s['filters']) for s in chart_data['series']]
        }
        # See of a user has overriden some of the values.
        if args.chart_opts:
            user_chart_opts = json.loads(args.chart_opts)
            for opt_name in ['title', 'xlabel', 'ylabel', 'legend']:
                if opt_name in user_chart_opts:
                    chart_opts[opt_name] = user_chart_opts[opt_name]
        # Plot the chart
        plt.title(chart_opts['title'])
        plt.xlabel(chart_opts['xlabel'])
        plt.ylabel(chart_opts['ylabel'])
        if args.chart_type == 'line':
            plt.xticks(sorted(map(int, chart_data['xvals'])))
            plt.grid()
            for idx, series in enumerate(chart_data['series']):
                xvals = sorted(map(int, series['data'].keys()))
                yvals = [series['data'][str(x)] for x in xvals]
                plt.plot(xvals, yvals, 'o-')
        else:
            ind = np.arange(3)
            plt.xticks(ind + 0.15, ('1', '2', '4'))
            for idx, series in enumerate(chart_data['series']):
                plt.bar(ind+0.1*idx, series['data'].values(), 0.1)
        plt.legend(chart_opts['legend'])
        plt.savefig(args.chart_file)


def main():
    """Entry point when invoking this scrip from a command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'inputs', nargs='*',
        help='Log directory or a JSON file'
    )
    parser.add_argument(
        '--recursive', required=False, default=False, action='store_true',
        help='If input is folder, scan it recursively for log files.'
    )
    parser.add_argument(
        '--xparam', type=str, required=True, default=None,
        help='A parameter that is associated with x axis.'
    )
    parser.add_argument(
        '--yparam', type=str, required=True, default=None,
        help='A parameter that is associated with y axis.'
    )
    parser.add_argument(
        '--series', type=str, required=True, default=None,
        help='A json array with filters for series.'
    )
    parser.add_argument(
        '--aggregation', type=str, required=True, default="avg",
        help='In case of multiple matches, use this to aggregate values (min, max, avg)'
    )
    parser.add_argument(
        '--chart_file', '--chart-file', type=str, required=False, default=None,
        help='If present, write chart into this file.'
    )
    parser.add_argument(
        '--series_file', '--series-file', type=str, required=False, default=None,
        help='If present, write series JSON data into this file.'
    )
    parser.add_argument(
        '--chart_opts', '--chart-opts', type=str, required=False, default=None,
        help='If present, a json object specifying chart options.'
    )
    parser.add_argument(
        '--chart_type', '--chart-type', type=str, required=False, default='line',
        help='Type of a chart ("line" or "bar").'
    )
    parser.add_argument(
        '--baseline_xvalue', '--baseline-xvalue', type=str, required=False, default=None,
        help="A value that's used to normalize one series. Useful to plot speedup charts."
    )
    parser.add_argument(
        '--baseline_series', '--baseline-series', type=int, required=False, default=None,
        help="An index of a baseline series to use to normalize all series."
    )
    args = parser.parse_args()

    if len(args.inputs) == 0:
        raise ValueError("Must be at least one input ('--input')")

    # Parse log files and load benchmark data
    logfiles = []      # Original raw log files with benchmark data
    benchmarks = []    # Parsed benchmarks
    for input_path in args.inputs:
        if os.path.isdir(input_path):
            logfiles.extend(IOUtils.find_files(input_path, "*.log", args.recursive))
        elif os.path.isfile(input_path) and input_path.endswith(('.json', '.json.gz')):
            file_benchmarks = IOUtils.read_json(input_path)
            if 'data' in file_benchmarks and isinstance(file_benchmarks['data'], list):
                benchmarks.extend(file_benchmarks['data'])
            else:
                logging.warn("Cannot parse file (%s). Invalid content.", input_path)
        else:
            logging.warn("Cannot parse file (%s). Unknown extension. ", input_path)
    if len(logfiles) > 0:
        benchmarks.extend(LogParser.parse_log_files(logfiles))
    else:
        logging.warn("No input log files have been found")
    if len(benchmarks) == 0:
        raise ValueError("No benchmarks have been loaded.")
    # Build data for series
    chart_data = SeriesBuilder.build(benchmarks, args)
    # Write it
    if args.series_file:
        DictUtils.dump_json_to_file(chart_data, args)
    # Plot it
    if args.chart_file:
        SeriesBuilder.plot(chart_data, args)


if __name__ == '__main__':
    main()
