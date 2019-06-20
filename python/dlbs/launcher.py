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
""" The launcher runs experiments one at a time.

It determines the benchmark backend launcher, builds its command line arguments, dumps all variables to a log file
and runs a benchmark.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os.path import isfile
import os
import sys
import copy
import logging
import datetime
import signal
from dlbs.worker import Worker
from dlbs.utils import DictUtils, ResourceMonitor, ParamUtils
from dlbs.exceptions import LogicError
from dlbs.bench_data import BenchData


class ProgressTracker(object):
    """Is used to track and report progress by writing progress information into a JSON file.

    That JSON file can be read by external tools (DLBS provides a simple web server as an example).
    """
    def __init__(self, num_experiments, num_active_experiments, file_name=None):
        """ Initialize progress reporter.

        Args:
            num_experiments (int): Total number of benchmarks in a plan.
            num_active_experiments (int): Number of benchmarks to run. May not equal to `num_experiments` due to
                multiple reasons. For instance, some benchmarks may be disabled.
            file_name (str): A file name to write progress information.
        """
        self.__file_name = file_name
        self.__progress = {
            'start_time': datetime.datetime.now(),
            'end_time': None,
            'status': 'inprogress',
            'num_total_benchmarks': num_experiments,
            'num_active_benchmarks': num_active_experiments,
            'num_completed_benchmarks': 0,
            'num_failed_benchmarks': 0,
            'num_successful_benchmarks': 0,
            'num_inactive_benchmarks': num_experiments - num_active_experiments,
            'num_existing_failed_benchmarks': 0,
            'num_existing_successful_benchmarks': 0,
            'active_benchmark': {},
            'completed_benchmarks': []
        }

    def num_completed_benchmarks(self):
        return self.__progress['num_completed_benchmarks']

    def print_summary(self):
        d = self.__progress
        num_existing_benchmarks = d['num_existing_successful_benchmarks'] + d['num_existing_failed_benchmarks']
        print("---------------------------------------------------------------------")
        print("- DLBS benchmark session summary (this is not a performance report) -")
        print("---------------------------------------------------------------------")
        print("  Start time: ................... {}".format(d['start_time']))
        print("  End time: ..................... {}".format(d['end_time']))
        print("  Duration (minutes): ........... {}".format((d['end_time'] - d['start_time']).total_seconds() / 60))
        print("  Total benchmarks in plan: ..... {}".format(d['num_total_benchmarks']))
        print("  |--Inactive benchmarks: ....... {}".format(d['num_inactive_benchmarks']))
        print("  |--Existing benchmarks: ....... {}".format(num_existing_benchmarks))
        print("  |  |--Successful benchmarks: .. {}".format(d['num_existing_successful_benchmarks']))
        print("  |  |--Failed benchmarks: ...... {}".format(d['num_existing_failed_benchmarks']))
        print("  |--Active benchmarks: ......... {}".format(d['num_active_benchmarks']))
        print("  |  |--Completed benchmarks: ... {}".format(d['num_completed_benchmarks']))
        print("  |  |--Successful benchmarks: .. {}".format(d['num_successful_benchmarks']))
        print("  |  |--Failed benchmarks: ...... {}".format(d['num_failed_benchmarks']))
        print("---------------------------------------------------------------------")
        print("- Your next steps: analyze performance data with bench_data.py      -")
        print("---------------------------------------------------------------------")

    def report(self, log_file, exec_status, bench_status=None):
        """ Report progress on done benchmark for 'disabled' or 'skipped' benchmarks.

        Args:
            log_file (str): A log file of the done benchmark.
            exec_status(str): An execution status of a benchmark:
                - 'inactive' User has decided not to run this benchmark.
                - 'skipped' Benchmark for which its log file has been found, benchmark has not been conducted.
                - 'completed' Benchmark has been conducted.
            bench_status (str): Benchmark status like ok or failure. May be None.
        """
        # 1. Verify that we have here expected execution status.
        expected_statuses = ['inactive', 'skipped', 'completed']
        if exec_status not in expected_statuses:
            raise LogicError("Unexpected benchmark execution status (={}'). "
                             "Expecting one of {}.".format(exec_status, expected_statuses))

        # 2. Update benchmark status.
        #    For benchmarks for which we are supposed to have log files, check its runtime status. In theory,
        #    `inactive` benchmarks may also have log files, but user has explicitly disabled such benchmarks.
        if exec_status == 'inactive':
            bench_status = None
        elif bench_status is None:
            # Either benchmark has been run or log file has been found. I do not rely on `exp.status` parameter in log
            # files because post-processing now is not good enough.
            bench_status = BenchData.status(log_file)

        # 3. Track information on this benchmark in a history array.
        if exec_status in ['inactive', 'skipped']:
            bench_info = {'start_time': datetime.datetime.now(),
                          'log_file': log_file}
        else:
            bench_info = self.__progress['active_benchmark']
            self.__progress["active_benchmark"] = {}
        bench_info.update({
            'end_time': datetime.datetime.now(),
            'exec_status': exec_status,
            'status': bench_status
        })
        self.__progress['completed_benchmarks'].append(bench_info)

        # 4. Update statistics
        if exec_status == 'inactive':
            self.__progress['num_inactive_benchmarks'] += 1
        elif exec_status == 'skipped':
            if bench_status == 'ok':
                self.__progress['num_existing_successful_benchmarks'] += 1
            else:
                self.__progress['num_existing_failed_benchmarks'] += 1
        else:
            self.__progress['num_completed_benchmarks'] += 1
            if bench_status == 'ok':
                self.__progress['num_successful_benchmarks'] += 1
            else:
                self.__progress['num_failed_benchmarks'] += 1

        # 5. If progress file has been provided, update that.
        if self.__file_name:
            DictUtils.dump_json_to_file(self.__progress, self.__file_name)

    def report_active(self, log_file):
        """ Report that new active benchmark has just started.

        Args:
            log_file (str): A log file for a currently active benchmark.
        """
        self.__progress['active_benchmark'] = {
            'exec_status': 'inprogress',
            'status': None,
            'start_time': datetime.datetime.now(),
            'end_time': None,
            'log_file': log_file
        }
        if self.__file_name:
            DictUtils.dump_json_to_file(self.__progress, self.__file_name)

    def report_all_completed(self):
        """Report all benchmarks have been done."""
        self.__progress['end_time'] = datetime.datetime.now()
        self.__progress['status'] = 'completed'
        if self.__file_name:
            DictUtils.dump_json_to_file(self.__progress, self.__file_name)


class Launcher(object):
    """Launcher runs benchmarks."""

    # Users can send -USR1 signal to request shutdown. In this case, this variable becomes True.
    must_exit = False

    @staticmethod
    def force_rerun(exp):
        """Does this experiment need to be re-run?

        By default, experiment is not ran if log file for this experiment exists. This does not work when part of
        the file path uses ${exp.id} value since it is generated each time.

        Args:
            exp (dict): Parameters of current experiment.

        Returns:
            bool: True if this benchmark needs to be run again even if its log file exists.
        """
        return 'exp.rerun' in exp and exp['exp.rerun'] is True

    @staticmethod
    def run(plan, progress_file=None):
        """Runs experiments in `plan` one at a time.

        In newest versions of this class the `plan` array must contain experiments with computed variables.

        Args:
            plan (list): List of benchmarks to perform (list of dictionaries).
            progress_file (str): A file for a progress reporter. If None, no progress will be reported.
        """
        num_experiments = len(plan)
        # See if resource monitor needs to be run. Now, the assumption is that
        # if it's enabled for a first experiments ,it's enabled for all others.
        resource_monitor = None
        if num_experiments > 0 and DictUtils.get(plan[0], 'monitor.frequency', 0) > 0:
            if not os.path.isdir(plan[0]['monitor.pid_folder']):
                os.makedirs(plan[0]['monitor.pid_folder'])
            resource_monitor = ResourceMonitor(
                plan[0]['monitor.launcher'], plan[0]['monitor.pid_folder'],
                plan[0]['monitor.frequency'], plan[0]['monitor.timeseries']
            )
            # The file must be created beforehand - this is required for docker to
            # to keep correct access rights.
            resource_monitor.empty_pid_file()
            resource_monitor.run()
        # It's used for reporting progress to a user
        num_active_experiments = 0
        for experiment in plan:
            if DictUtils.get(experiment, 'exp.status', '') not in ['disabled', 'inactive']:
                num_active_experiments += 1
        progress_tracker = ProgressTracker(num_experiments,
                                           num_active_experiments,
                                           progress_file)
        # Setting handler for SIGUSR1 signal. Users can send this signal to this
        # script to gracefully terminate benchmarking process.
        print("--------------------------------------------------------------")
        print("Experimenter pid %d. Run this to gracefully terminate me:" % os.getpid())
        print("\tkill -USR1 %d" % os.getpid())
        print("I will terminate myself as soon as current benchmark finishes.")
        print("--------------------------------------------------------------")
        sys.stdout.flush()
        Launcher.must_exit = False

        def _sigusr1_handler(signum, frame):
            Launcher.must_exit = True
        signal.signal(signal.SIGUSR1, _sigusr1_handler)

        for idx in range(num_experiments):
            if Launcher.must_exit:
                logging.warn(
                    "The SIGUSR1 signal has been caught, gracefully shutting down benchmarking "
                    "process on experiment %d (out of %d)",
                    idx,
                    num_experiments
                )
                break
            experiment = plan[idx]
            # Is experiment disabled?
            if DictUtils.get(experiment, 'exp.status', '') in ('disabled', 'inactive'):
                logging.info("Will not run benchmark, reason: exp.status='%s'" % experiment['exp.status'])
                progress_tracker.report(experiment['exp.log_file'], exec_status='inactive')
                continue
            # If experiments have been ran, check if we need to re-run.
            if DictUtils.get(experiment, 'exp.log_file', None) is not None:
                if isfile(experiment['exp.log_file']):
                    bench_status = None
                    no_rerun_msg = None
                    rerun_condition = DictUtils.get(experiment, 'exp.rerun', 'never')
                    if rerun_condition == 'never':
                        no_rerun_msg = "Will not run benchmark, reason: log file exists, exp.rerun='never'"
                    elif rerun_condition == 'onfail':
                        bench_status = BenchData.status(experiment['exp.log_file'])
                        if bench_status == 'ok':
                            no_rerun_msg = "Will not run benchmark, reason: log file exists, exp.status='ok', "\
                                           "exp.rerun='onfail'"
                    if no_rerun_msg is not None:
                        logging.info(no_rerun_msg)
                        progress_tracker.report(experiment['exp.log_file'], exec_status='skipped',
                                                bench_status=bench_status)
                        continue
            # Track current progress
            progress_tracker.report_active(DictUtils.get(experiment, 'exp.log_file', '<none>'))
            # Get script that runs experiment for this framework. If no 'framework_family' is
            # found, we can try to use exp.framework.
            framework_key = 'exp.framework_family'
            if framework_key not in experiment:
                framework_key = 'exp.framework'
            command = [experiment['%s.launcher' % (experiment[framework_key])]]
            # Do we need to manipulate arguments for launching process?
            launcher_args_key = '%s.launcher_args' % experiment[framework_key]
            if launcher_args_key in experiment:
                launcher_args = set(experiment[launcher_args_key].split(' '))
                logging.debug(
                    'Only these arguments will be passed to launching process (%s): %s',
                    command[0],
                    str(launcher_args)
                )
            else:
                launcher_args = None
            for param, param_val in experiment.items():
                if launcher_args is not None and param not in launcher_args:
                    continue
                if isinstance(param_val, list):
                    raise ValueError("Here, this must not be the list but (%s=%s)" % (param, str(param_val)))
                if not isinstance(param_val, bool):
                    command.extend(['--%s' % (param.replace('.', '_')), ParamUtils.to_string(param_val)])
                else:
                    command.extend(['--%s' % (param.replace('.', '_')), ('true' if param_val else 'false')])
            # Prepare environmental variables
            env_vars = copy.deepcopy(os.environ)
            env_vars.update(DictUtils.filter_by_key_prefix(
                experiment,
                'runtime.env.',
                remove_prefix=True
            ))
            # Run experiment in background and wait for complete
            worker = Worker(command, env_vars, experiment)
            worker.work(resource_monitor)
            # Print progress
            progress_tracker.report(experiment['exp.log_file'], exec_status='completed')
            if progress_tracker.num_completed_benchmarks() % 10 == 0:
                print("Done %d benchmarks out of %d" % (progress_tracker.num_completed_benchmarks(),
                                                        num_active_experiments))
                sys.stdout.flush()
        # All benchmarks have been conducted.
        if resource_monitor is not None:
            resource_monitor.stop()
        progress_tracker.report_all_completed()
        progress_tracker.print_summary()
