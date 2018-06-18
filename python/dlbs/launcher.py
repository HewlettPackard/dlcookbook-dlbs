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

It determines the framework launcher, builds its command line arguments,
dumps all variables to log file and runs experiment.
"""
from __future__ import print_function
from os.path import isfile
import os
import sys
import copy
import logging
import datetime
import json
import signal
import uuid
import shutil
from dlbs.worker import Worker
from dlbs.utils import DictUtils
from dlbs.utils import ResourceMonitor
from dlbs.utils import param2str

class ProgressReporter(object):
    def __init__(self, num_experiments, num_active_experiments, file_name=None):
        self.__file_name = file_name
        if self.__file_name:
            self.__progress = {
                'start_time': str(datetime.datetime.now()),
                'stop_time': None,
                'status': 'inprogress',
                'num_total_benchmarks': num_experiments,
                'num_active_benchmarks': num_active_experiments,
                'num_completed_benchmarks': 0,
                'active_benchmark': {},
                'completed_benchmarks':[]
            }


    def report(self, log_file, status, counts=True):
        if self.__file_name:
            self.__progress['completed_benchmarks'].append({
                'status': status,
                'start_time': str(datetime.datetime.now()),
                'stop_time': str(datetime.datetime.now()),
                'log_file': log_file
            })
            if counts:
                self.__progress['num_completed_benchmarks'] += 1
            DictUtils.dump_json_to_file(self.__progress, self.__file_name)

    def report_active(self, log_file):
        if self.__file_name:
            self.__progress['active_benchmark'] = {
                'status': 'inprogress',
                'start_time': str(datetime.datetime.now()),
                'stop_time': None,
                'log_file': log_file
            }
            DictUtils.dump_json_to_file(self.__progress, self.__file_name)

    def report_active_completed(self):
        if self.__file_name:
            self.__progress['active_benchmark']['stop_time'] = str(datetime.datetime.now())
            self.__progress['active_benchmark']['status'] = 'completed'
            self.__progress['completed_benchmarks'].append(self.__progress['active_benchmark'])
            self.__progress['num_completed_benchmarks'] += 1
            self.__progress["active_benchmark"] = {}
            DictUtils.dump_json_to_file(self.__progress, self.__file_name)

    def report_all_completed(self):
        if self.__file_name:
            self.__progress['stop_time'] = str(datetime.datetime.now())
            self.__progress['status'] = 'completed'
            DictUtils.dump_json_to_file(self.__progress, self.__file_name)


class Launcher(object):
    """Launcher runs experiments."""

    must_exit = False
    uuid=str(uuid.uuid4()).replace('-','')

    @staticmethod
    def force_redo(exp):
        """Does this experiment need to be re-run?

        By default, experiment is not ran if log file for this experiment exists.
        This does not work when part of the file path uses ${exp.id} value since
        it is generated each time.

        :param dict exp: Parameters of current experiment.
        :return: True if experiment needs to be re-run, False otherwise.
        :rtype: bool
        """
        return 'exp.rerun' in exp and exp['exp.rerun'] is True

    @staticmethod
    def run(plan, progress_file=None):
        """Runs experiments.

        In newest versions of this class the **plan** array must contain experiments
        with computed variables.

        :param list plan: List of experiments to perform.
        """
        # Count number of active experiments in the plan
        num_active_experiments = 0
        for experiment in plan:
            if 'exp.status' in experiment and experiment['exp.status'] != 'disabled':
                num_active_experiments += 1

        num_experiments = len(plan)
        start_time = datetime.datetime.now()
        stats = {
            "launcher.total_experiments": num_experiments,
            "launcher.active_experiments": num_active_experiments,
            "launcher.failed_experiments": 0,
            "launcher.skipped_experiments": 0,
            "launcher.disabled_experiments": 0,
            "launcher.start_time": str(start_time)
        }
        # See if resource monitor needs to be run. Now, the assumption is that
        # if it's enabled for a first experiments ,it's enabled for all others.
        resource_monitor = None
        if num_experiments > 0 and 'monitor.frequency' in plan[0] and plan[0]['monitor.frequency'] > 0:
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
        progress_reporter = ProgressReporter(
            num_experiments,
            num_active_experiments,
            progress_file
        )
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
        num_completed_experiments = 0
        for idx in range(num_experiments):
            if Launcher.must_exit:
                logging.warn(
                    "The SIGUSR1 signal has been caught, gracefully shutting down benchmarking process on experiment %d (out of %d)",
                    idx,
                    num_experiments
                )
                break
            experiment = plan[idx]
            # Is experiment disabled?
            if 'exp.status' in experiment and experiment['exp.status'] == 'disabled':
                logging.info("Disabling experiment, exp.disabled is true")
                stats['launcher.disabled_experiments'] += 1
                progress_reporter.report(experiment['exp.log_file'], 'disabled', counts=False)
                continue
            # If experiments have been ran, check if we need to re-run.
            if 'exp.log_file' in experiment and experiment['exp.log_file']:
                if isfile(experiment['exp.log_file']) and not Launcher.force_redo(experiment):
                    logging.info(
                        "Skipping experiment, file (%s) exists",
                        experiment['exp.log_file']
                    )
                    stats['launcher.skipped_experiments'] += 1
                    progress_reporter.report(experiment['exp.log_file'], 'skipped', counts=True)
                    continue
            # Track current progress
            progress_reporter.report_active(experiment['exp.log_file'])
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
                    'Only these arguments will be passed to laucnhing process (%s): %s',
                    command[0],
                    str(launcher_args)
                )
            else:
                launcher_args = None
            for param, param_val in experiment.items():
                if launcher_args is not None and param not in launcher_args:
                    continue
                assert not isinstance(param_val, list),\
                       "Here, this must not be the list but (%s=%s)" % (param, str(param_val))
                if not isinstance(param_val, bool):
                    command.extend(['--%s' % (param.replace('.', '_')), param2str(param_val)])
                else:
                    command.extend(['--%s' % (param.replace('.', '_')), ('true' if param_val else 'false')])
            # Prepare environmental variables
            env_vars = copy.deepcopy(os.environ)
            env_vars.update(DictUtils.filter_by_key_prefix(
                experiment,
                'runtime.env.',
                remove_prefix=True
            ))
            # Run experiment in background and wait for completion
            worker = Worker(command, env_vars, experiment)
            worker.work(resource_monitor)
            if worker.ret_code != 0:
                stats['launcher.failed_experiments'] += 1
            num_completed_experiments += 1
            # Print progress
            if num_completed_experiments%10 == 0:
                print("Done %d benchmarks out of %d" % (num_completed_experiments, num_active_experiments))
            progress_reporter.report_active_completed()

        try:
            shutil.rmtree(plan[0]['runtime.cuda_cache'], ignore_errors=True)
        except IOError:
            pass
        end_time = datetime.datetime.now()
        stats['launcher.end_time'] = str(end_time)
        stats['launcher.hours'] = (end_time - start_time).total_seconds() / 3600

        if resource_monitor is not None:
            resource_monitor.stop()

        for key, val in stats.items():
            print('__%s__=%s' % (key, json.dumps(val)))
        progress_reporter.report_all_completed()
