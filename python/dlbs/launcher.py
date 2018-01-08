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
import copy
import logging
import datetime
import json
from dlbs.processor import Processor
from dlbs.worker import Worker
from dlbs.utils import DictUtils
from dlbs.utils import ResourceMonitor
from dlbs.utils import IOUtils

class Launcher(object):
    """Launcher runs experiments."""

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
        return 'exp.force_rerun' in exp and exp['exp.force_rerun'] == 'true'

    @staticmethod
    def run(plan, compute_variables=True):
        """Runs experiments.

        Before running experiments, make sure we can compute variables. Exception
        will be thrown and no experiments will be run. Compute variable will
        convert all lists into white space separated strings.

        :param list plan: List of experiments to perform.
        :param bool compute_variables: If true, variables need to be comptued.
        """
        if compute_variables:
            Processor().compute_variables(plan)
        num_experiments = len(plan)
        start_time = datetime.datetime.now()
        stats = {
            "launcher.total_experiments": num_experiments,
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

        for idx in range(num_experiments):
            experiment = plan[idx]
            # Is experiment disabled?
            if 'exp.disabled' in experiment and experiment['exp.disabled'] == 'true':
                logging.info("Disabling experiment, exp.disabled is true")
                stats['launcher.disabled_experiments'] += 1
                continue
            # If experiments have been ran, check if we need to re-run.
            if 'exp.log_file' in experiment and experiment['exp.log_file']:
                if isfile(experiment['exp.log_file']) and not Launcher.force_redo(experiment):
                    logging.info(
                        "Skipping experiment, file (%s) exists",
                        experiment['exp.log_file']
                    )
                    stats['launcher.skipped_experiments'] += 1
                    continue
            # Get script that runs experiment for this framework
            command = [experiment['%s.launcher' % experiment['exp.framework']]]
            # Do we need to manipulate arguments for launching process?
            launcher_args_key = '%s.launcher.args' % experiment['exp.framework']
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
                command.extend(['--%s' % (param.replace('.', '_')), str(param_val)])
            # Prepare environmental variables
            env_vars = copy.deepcopy(os.environ)
            env_vars.update(DictUtils.filter_by_key_prefix(
                experiment,
                'runtime.env.',
                remove_prefix=True
            ))
            # Run experiment in background and wait for complete
            worker = Worker(command, env_vars, experiment, idx+1, num_experiments)
            worker.work(resource_monitor)
            if worker.ret_code != 0:
                stats['launcher.failed_experiments'] += 1

        end_time = datetime.datetime.now()
        stats['launcher.end_time'] = str(end_time)
        stats['launcher.hours'] = (end_time - start_time).total_seconds() / 3600

        if resource_monitor is not None:
            resource_monitor.stop()

        for key, val in stats.items():
            print('__%s__=%s' % (key, json.dumps(val)))
