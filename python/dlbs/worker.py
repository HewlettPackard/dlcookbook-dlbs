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
""":py:class:`~dlbs.Worker` class runs one benchmarking experiment."""
import sys
import time
import json
import threading
import logging
import subprocess
import traceback
from dlbs.utils import IOUtils
from dlbs.utils import DictUtils
from dlbs.sysinfo.systemconfig import SysInfo

class Worker(threading.Thread):
    """This class runs one benchmarking experiment.

    It runs it in a separate thread. The typical usage example:

    .. code-block:: python

        worker = Worker(
            ["echo", "'Hello World'"], # Command to execute (with ``Popen``)
            {},                        # Environmental variables to set
            {},                        # Experiment variables (dictionary)
            5,                         # Index of current benchmark
            10                         # Total number of benchmarks
        )
        # This is a blocking call.
        worker.work()
    """
    def __init__(self, command, environ, params):
        """ Initializes this worker with the specific parameters.

        :param list command: List containing command to execute and its comamnd\
                             line arguments (with Popen).
        :param dict environ: Environment variables to set with Popen.
        :param dict params: Parameters of this experiment (dictionary).

        """
        threading.Thread.__init__(self)
        self.command = command            # Command + command line arguments
        self.environ = environ            # Environmental variables to append
        self.params = params              # All experiment variables
        self.process = None               # Background process object
        self.ret_code = 0                 # Return code of the process

    def __dump_parameters(self, a_file):
        """Dumps all experiment parameters to a file (or /dev/stdout)."""
        a_file.write("Running subprocess (%s) with log file '%s'\n" % (self.command, self.params['exp.log_file']))
        a_file.write("\n-----------------------------------\nVariables from python entry script.\n-----------------------------------\n")
        for key, val in self.params.items():
            a_file.write('__%s__=%s\n' % (key, json.dumps(val)))
        a_file.write("\n----------------------------\nStarting framework launcher.\n----------------------------\n")

    def run(self):
        """Runs subprocess with Popen.

        This method must not be called directly. Use blocking :py:meth:`~dlbs.Worker.work`
        method instead.
        """
        try:
            # Dump parameters to a log file or to standard output
            DictUtils.ensure_exists(self.params, 'exp.log_file', default_value='')
            if self.params['exp.log_file'].strip() == '':
                self.params['exp.log_file'] = '/dev/stdout'
            IOUtils.mkdirf(self.params['exp.log_file'])
            with open(self.params['exp.log_file'], 'a+') as log_file:
                self.__dump_parameters(log_file)
            # This is where we launch process. Keep in mind, that the log file that's
            # supposed to be created is exp.log_file or exp_log_file in the script.
            # Other output of the launching script will be printed by this pyhton code
            # to a stanard output.
            self.process = subprocess.Popen(self.command, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=self.environ)
            while True:
                output = self.process.stdout.readline()
                if output == '' and self.process.poll() is not None:
                    break
                if output:
                    sys.stdout.write(output)
                    sys.stdout.flush()
            self.ret_code = self.process.poll()
        except Exception as err:
            logging.warn('Exception has been caught for experiment %s: %s', self.params.get('exp.id'), str(err))
            logging.warn(traceback.format_exc())
            self.ret_code = -1

    def work(self, resource_monitor):
        """Runs experiment as subprocess and waits for its completion.

        :return: Status code.
        """
        self.start()
        self.join()
        if resource_monitor is not None:
            resource_monitor.empty_pid_file()
            metrics = resource_monitor.get_measurements()
            with open(self.params['exp.log_file'], 'a+') as log_file:
                for key in metrics:
                    log_file.write('__results.use.%s__=%s\n' % (key, json.dumps(metrics[key])))
        if self.is_alive():
            self.process.terminate()
            self.join()
            time.sleep(1)
        if 'exp.sys_info' in self.params and self.params['exp.sys_info']:
            info = SysInfo(self.params['exp.sys_info']).collect()
            with open(self.params['exp.log_file'], 'a+') as log_file:
                for key in info:
                    log_file.write('__%s__=%s\n' % (key, json.dumps(info[key])))
        return self.ret_code
