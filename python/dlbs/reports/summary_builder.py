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
"""The module builds three types of reports that are readable by humans. It also
generates json files that are used to generate data for HPE Discover Demo.

Usage:

>>> python summary_builder.py [PARAMETERS]

Parameters:

* ``--summary-file`` File name (json) with experiment results. This file is produced
  by a log parser.
* ``--report-file`` File name of the report to be generated.
* ``--type`` Type of the report ('exploration', 'weak-scaling', 'strong-scaling')
* ``--target-variable`` Target variable for the report. In most cases it's either
  'results.training_time' or 'results.inference_time'.
* ``--query`` Optional JSON flat dictionary. Specifies query that selects experiments
  to build summary for. A typical use case is to select specific framework. For instance:
  **--query='{\"exp.framework_id\": \"tensorflow\"}'**. Should be json parsable string.
"""
from __future__ import print_function
import json
import argparse
from sets import Set
import dlbs.python_version   # pylint: disable=unused-import
from dlbs.utils import DictUtils, OpenFile


BATCH_TM_TITLE = "Batch time (milliseconds)"
IPS_TITLE = "Inferences Per Second (IPS, throughput)"
SPEEDUP_TITLE = "Speedup (instances per second)"

class SummaryBuilder(object):
    """Class that builds summary reports in csv formats and generates json files."""

    def __init__(self):
        self.cache = None
        self.nets = None
        self.batches = None
        self.devices = None

    def build_cache(self, summary_file, target_variable, query):
        """Loads data from json file."""
        with OpenFile(summary_file) as file_obj:
            summary = json.load(file_obj)
        self.cache = {}
        self.nets = Set()
        self.batches = Set()
        self.devices = Set()
        for experiment in summary['data']:
            if target_variable not in experiment:
                print("target variable not in experiment, skipping")
                continue
            if not DictUtils.match(experiment, query, policy='strict'):
                continue
            # batch is an effective batch here
            key = '{0}_{1}_{2}'.format(
                experiment['exp.model_title'],
                experiment['exp.gpus'],
                experiment['exp.effective_batch']
            )
            self.cache[key] = float(experiment[target_variable])
            self.nets.add(experiment['exp.model_title'])
            self.batches.add(int(experiment['exp.effective_batch']))
            self.devices.add(str(experiment['exp.gpus']))
        self.nets = sorted(list(self.nets))
        self.batches = sorted(list(self.batches))
        self.devices = sorted(list(self.devices), key=len)

    # 1. Batch time in milliseconds
    # 2. Instance Rate
    # 3. Performance relative to first batch
    def build_exploration_report(self, report_file):
        """ Builds exploration report for inference and single device training.
        """
        header = "%-20s %-10s" % ('Network', 'Device')
        for batch in self.batches:
            header = "%s %-10s" % (header, batch)
        report = []
        json_report = {'data': []}
        for net in self.nets:
            for device in self.devices:
                profile = {
                    'net': net,
                    'device': device,
                    'time': [],
                    'throughput': []
                }
                profile_ok = False
                for batch in self.batches:
                    key = '{0}_{1}_{2}'.format(net, device, batch)
                    batch_tm = throughput = -1
                    if key in self.cache and self.cache[key] > 0:
                        batch_tm = self.cache[key]
                        profile_ok = True
                        throughput = int(batch * (1000.0 / batch_tm))
                        json_profile = SummaryBuilder.default_json_profile(net, 'strong', batch)
                        json_profile['perf']['data']['1'] = batch_tm
                        json_report['data'].append(json_profile)
                    profile['time'].append(round(batch_tm, 3))
                    profile['throughput'].append(throughput)
                if profile_ok:
                    report.append(profile)
        SummaryBuilder.print_report_txt(BATCH_TM_TITLE, header, report, 'net', 'device', 'time')
        SummaryBuilder.print_report_txt(IPS_TITLE, header, report, 'net', 'device', 'throughput')
        DictUtils.dump_json_to_file(json_report, report_file)

    # Assuming that the first device in a list is a single GPU (CPU) device.
    def build_strong_scaling_report(self, jsonfile):
        """ Builds strong scaling report for multi-GPU training.
        """
        header = "%-20s %-10s" % ('Network', 'Batch')
        for device in self.devices:
            header = "%s %-10s" % (header, (1 + device.count(',')))
        report = []
        json_report = {'data': []}
        for net in self.nets:
            for batch in self.batches:
                profile = {
                    'net': net,
                    'batch': batch,
                    'time': [],
                    'throughput': [],
                    'efficiency': [],
                    'speedup': []
                }
                json_profile = SummaryBuilder.default_json_profile(net, 'strong', batch)
                profile_ok = False
                # device here is '0', '0,1', '0,1,2,3' ...
                for device in self.devices:
                    key = '{0}_{1}_{2}'.format(net, device, batch)
                    batch_tm = throughput = efficiency = speedup = -1
                    num_devices = 1 + device.count(',')
                    if key in self.cache:
                        batch_tm = self.cache[key]
                        throughput = int(batch * (1000.0 / batch_tm))
                        json_profile['perf']['data'][str(num_devices)] = batch_tm
                        if len(profile['throughput']) == 0:
                            speedup = 1
                        else:
                            speedup = 1.0 * throughput / profile['throughput'][0]
                    if len(profile['efficiency']) == 0:
                        efficiency = 100.00
                        profile_ok = True
                    elif profile['time'][0] > 0:
                        efficiency = int(10000.0 * profile['time'][0] / (num_devices * batch_tm))/100.0
                        profile_ok = True
                    profile['time'].append(batch_tm)
                    profile['throughput'].append(throughput)
                    profile['efficiency'].append(efficiency)
                    profile['speedup'].append(speedup)
                if profile_ok:
                    report.append(profile)
                    json_report['data'].append(json_profile)
        SummaryBuilder.print_report_txt(BATCH_TM_TITLE, header, report, 'net', 'batch', 'time')
        SummaryBuilder.print_report_txt(IPS_TITLE, header, report, 'net', 'batch', 'throughput')
        SummaryBuilder.print_report_txt(SPEEDUP_TITLE, header, report, 'net', 'batch', 'speedup')
        SummaryBuilder.print_report_txt(
            "Efficiency = 100% * t1 / (N * tN)",
            header, report, 'net', 'batch', 'efficiency'
        )
        DictUtils.dump_json_to_file(json_report, jsonfile)


    # Assuming that the first device in a list is a single GPU (CPU) device.
    def build_weak_scaling_report(self, jsonfile):
        """ Builds weak scaling report for multi-GPU training.
        """
        header = "%-20s %-10s" % ('Network', 'Batch')
        for device in self.devices:
            header = "%s %-10d" % (header, (1 + device.count(',')))
        report = []
        json_report = {'data': []}
        for net in self.nets:
            for batch in self.batches:
                # batch is the base 'batch size' i.e. for a one GPU
                profile = {
                    'net': net,         # network name
                    'batch': batch,     # per device batch size
                    'time': [],         # batch times
                    'throughput': [],   # throughput
                    'efficiency': [],   # efficiency
                    'speedup': []       # speedup
                }
                json_profile = SummaryBuilder.default_json_profile(net, 'weak', batch)
                profile_ok = False
                for device in self.devices:
                    # weak scaling: we want to find results for effective batch size
                    # which is N * batch
                    num_devices = 1 + device.count(',')
                    key = '{0}_{1}_{2}'.format(net, device, (batch*num_devices))
                    if num_devices == 1 and key not in self.cache:
                        # If we do not have data for one device, does not make sense
                        # to continue
                        break
                    batch_tm = throughput = efficiency = speedup = -1.0
                    if key in self.cache:
                        batch_tm = self.cache[key]
                        throughput = int((num_devices*batch) * (1000.0 / batch_tm))
                        json_profile['perf']['data'][str(num_devices)] = batch_tm
                        if len(profile['throughput']) == 0:
                            speedup = 1
                        else:
                            speedup = 1.0 * throughput / profile['throughput'][0]
                    if len(profile['efficiency']) == 0:
                        efficiency = 100.00
                        profile_ok = True
                    elif profile['time'][0] > 0:
                        efficiency = int(10000.0 * profile['time'][0] / batch_tm) / 100.0
                        profile_ok = True
                    profile['time'].append(batch_tm)
                    profile['throughput'].append(int(throughput))
                    profile['efficiency'].append(efficiency)
                    profile['speedup'].append(speedup)
                if profile_ok:
                    report.append(profile)
                    json_report['data'].append(json_profile)
        SummaryBuilder.print_report_txt(BATCH_TM_TITLE, header, report, 'net', 'batch', 'time')
        SummaryBuilder.print_report_txt(IPS_TITLE, header, report, 'net', 'batch', 'throughput')
        SummaryBuilder.print_report_txt(SPEEDUP_TITLE, header, report, 'net', 'batch', 'speedup')
        SummaryBuilder.print_report_txt(
            "Efficiency  = 100% * t1 / tN",
            header, report, 'net', 'batch', 'efficiency'
        )
        DictUtils.dump_json_to_file(json_report, jsonfile)

    @staticmethod
    def default_json_profile(net_name, scaling, batch_size):
        """ Returns an instance of data structure that used to store
            data in json format.
        """
        json_profile = {
            'm': net_name,
            'hw': {
                'name': '',
                'pu': '',
                'n': 1,
                'cluster': {
                    'sz': 1,
                    'interconnect': ''
                }
            },
            'sw': {
                'rtm': '',
                'bsz': batch_size,
                'btype': scaling
            },
            'perf': {
                'data': {},
                'estimates': []
            }
        }
        return json_profile

    @staticmethod
    def print_report_txt(description, header, report,
                         col1_key, col2_key, data_key):
        """ Writes a human readable report to a standard output.
        """
        print(description)
        print(header)
        for record in report:
            row = "%-20s %-10s" % (record[col1_key], record[col2_key])
            for idx in range(len(record['time'])):
                val = record[data_key][idx]
                if val >= 0:
                    if isinstance(val, int):
                        row = "%s %-10d" % (row, record[data_key][idx])
                    else:
                        row = "%s %-10.2f" % (row, record[data_key][idx])
                else:
                    row = "%s %-10s" % (row, '-')
            print(row)
        print("\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_file', '--summary-file', required=True, help="File name (json) with experiment results. This file is produced by a log parser.")
    parser.add_argument('--report_file', '--report-file', required=False, default=None, help="File name of the report to be generated.")
    parser.add_argument('--type', help="Type of the report ('exploration', 'weak-scaling', 'strong-scaling')")
    parser.add_argument('--target_variable', '--target-variable', help="Target variable for the report. In most cases it's 'results.time'.")
    parser.add_argument('--query', required=False, type=str, default="{}",
                                   help="Optional JSON flat dictionary. Specifies query that selects experiments to build summary for.\
                                         A typical use case is to select specific framework. For instance:\
                                         --query='{\"exp.framework\": \"tensorflow\"}'. Should be json parsable string")
    args = parser.parse_args()

    query = json.loads(args.query)
    summary_builder = SummaryBuilder()
    summary_builder.build_cache(args.summary_file, args.target_variable, query)
    builder_funcs = {
        'exploration': summary_builder.build_exploration_report,
        'strong-scaling': summary_builder.build_strong_scaling_report,
        'weak-scaling': summary_builder.build_weak_scaling_report
    }
    assert args.type in builder_funcs, "Invalid report type '%s'" % (args.type)
    builder_funcs[args.type](args.report_file)
