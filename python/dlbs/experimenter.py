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
"""The main entry point script for this project.

Experimenter takes as an input a specification of experiments, builds plan (just a
set of experiments to run) and runs experiments one at a time. It accepts the following
command line parameters::

    $ python experimenter.py ACTION [parameters]

* ``ACTION`` Action to perform. Valid actions are

  * **print-config**
  * **run** Build plan and run experiments.
  * **build** Only build plan. If file name specified, serialize to file.
  * **validate** Analyze plan - run several validation checks to make sure \
    operating system is properly tuned (see :py:class:`~dlbs.validator.Validator` class for details)

* Parameters

  * ``--config`` Configuration file (json) of an experiment. Will override values from default configuration.
  * ``--plan`` Pre-built plan of an experiment (json). If action is **build**, a file name to write plan to.\
    If action is **run**, a file name to read plan from.
  * ``-P`` Parameters that override parameters in configuration file. For instance, ``-Pexp.phase='"inference"'``.\
    Values must be json parsable (json.loads()).
  * ``-V`` Variables that override variables in configuration file in section "variables".\
    These variables are used to generate different combinations of experiments.\
    For instance: ``-Vexp.framework='["tensorflow", "caffe2"]'``. Values must be\
    json parsable (json.loads()).
  * ``--log-level`` Python logging level. Valid values: "critical", "error", "warning", "info" and "debug"
  * ``--discard-default-config`` Do not load default configuration.
  * ``-E`` Extensions to add. Can be usefull to quickly customize experiments. Must be valid json\
    parsable array element for "extension" array.

Example:
   Load default configuration, pretty print it to a command line and exit. Without other arguments,
   it will print default configuration. Parameters and variables defined in configuration files will
   not be evaluated. The 'print-config' just prints what's inside configuration files i.e. parameters/variables
   passed via comamnd line arguments will not be included::

       $ python experimenter.py print-config --log-level=debug

Example:
   There are two types of variables. The first type is **parameter** variables or just parameter.
   These parameters do not contribute to generating different experiments and may be common to all
   experiments. It's possible to specify them on a command line. All values of such paarmeters must be json
   parsable (json.loads())::

       $ python experimenter.py build --discard-default-config --log-level=debug  \\
       $                              -Pstr.greeting='"Hello World!"' -Pint.value=3 \\
       $                              -Pfloat.value=3.4343 -Plist.value='["1", "2", "3"]' \\
       $                              -Plist.value2='[100,101,102]'

Example:
   A minimal working example to run BVLC Caffe. Run one experiment and store results in a file.
   If you run multiple experiments, you really want to make sure that experiment log file is
   different for every experiment (assuming you run it from DLBS_ROOT/tutorials/dlcookbook)::

       $ export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
       $ export CUDA_CACHE_PATH=/dev/shm/cuda_cache
       $ . ${BENCH_ROOT}/../../scripts/environment.sh
       $ script=$DLBS_ROOT/python/dlbs/experimenter.py
       $
       $ python experimenter.py run --log-level=debug \\
       $                            -Pexp.framework='"bvlc_caffe"' \\
       $                            -Pexp.env='"docker"' \\
       $                            -Pexp.gpus='0' \\
       $                            -Pexp.model='"alexnet"' \\
       $                            -Pexp.device_batch='"16"'\\
       $                            -Pexp.log_file='"${BENCH_ROOT}/${caffe.fork}_caffe/training.log"'
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import logging
import argparse
import json
import copy
from dlbs.builder import Builder
from dlbs.launcher import Launcher
from dlbs.utils import DictUtils
from dlbs.utils import ConfigurationLoader
from dlbs.validator import Validator
from dlbs.processor import Processor
from dlbs.help.helper import Helper
from dlbs.sysinfo.systemconfig import SysInfo


class Experimenter(object):
    """Class that generates configurations and runs experiments"""

    ACTIONS = ['print-config', 'run', 'build', 'validate']

    def __init__(self):
        self.__validation = True         # Validate config before running benchmarks
        self.__action = None             # Action to perform (build, run, ...)
        self.__config_file = None        # Configuration file to load
        self.__progress_file = None      # A JSON file with current progress
        self.__config = {}               # Loaded configuration
        self.__param_info = {}           # Parameter meta-info such as type and value domain
        self.__plan_file = None          # File with pre-built plan
        self.__plan = []                 # Loaded or generated plan
        self.__params = {}               # Override env variables from files
        self.__variables = {}            # Override variables from files
        self.__initialized = False       # Experimenter can be initialized only once
        # Dirty hacks
        for var in ('CUDA_CACHE_PATH', 'http_proxy', 'https_proxy'):
            DictUtils.ensure_exists(os.environ, var, '')
        DictUtils.ensure_exists(
            os.environ,
            'DLBS_ROOT',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
        )

    @property
    def validation(self):
        """Do we need to perform validation."""
        return self.__validation

    @validation.setter
    def validation(self, validation):
        """Set validation."""
        self.__validation = validation

    @property
    def action(self):
        """Get current action."""
        return self.__action

    @action.setter
    def action(self, action):
        """Set current action."""
        if action not in Experimenter.ACTIONS:
            raise ValueError('Invalid value for action (%s). Must be one of %s' % (action, str(Experimenter.ACTIONS)))
        self.__action = action

    @property
    def config_file(self):
        """Get configuration file."""
        return self.__config_file

    @config_file.setter
    def config_file(self, config_file):
        """Set configuration file."""
        self.__config_file = config_file

    @property
    def config(self):
        """Get configuration."""
        return self.__config

    @config.setter
    def config(self, config):
        """Set configuration."""
        self.__config = config

    @property
    def param_info(self):
        """Get parameters info dictionary."""
        return self.__param_info

    @param_info.setter
    def param_info(self, param_info):
        """Set parameters info dictionary."""
        self.__param_info = param_info

    @property
    def plan_file(self):
        """Set plan file."""
        return self.__plan_file

    @plan_file.setter
    def plan_file(self, plan_file):
        """Get plan file."""
        self.__plan_file = plan_file

    @property
    def plan(self):
        """Get plan."""
        return self.__plan

    @plan.setter
    def plan(self, plan):
        """Set plan."""
        self.__plan = plan

    @property
    def params(self):
        """Get parameters."""
        return self.__params

    @property
    def variables(self):
        """Get variables."""
        return self.__variables

    def init(self, **kwargs):
        """Initializes experimenter.

        Args:
            **kwargs (dict): Optional initialization parameters:
                - action (str): Action to perform.
                - config (str): A user-provided configuration file.
                - plan (str): A file for generated benchmark plan.
                - no_validation (bool): If true, do not perform validation
                - progress_file (str): A path to progress file (if not None, enables progress reporting).
                - params (dict): User defined parameters.
                - vars (dict): User defined variables.
                - discard_default_config (bool): If True, do not load standard DLBS config.
                - extensions (dict): User provided extensions.

        User provided parameters (`params`), variables (`vars`) and extensions (`extensions`) overwrite values defined
        in user configuration files (`config`) if it is present.
        Information defined in a uses-provided configuration file (`config`) overwrites standard DLBS configuration.
        """
        if self.__initialized:
            raise RuntimeError("Experimenter can only be initialized once.")

        self.action = DictUtils.get(kwargs, 'action', 'run')
        self.config_file = DictUtils.get(kwargs, 'config', None)
        self.plan_file = DictUtils.get(kwargs, 'plan', None)
        self.validation = not DictUtils.get(kwargs, 'no_validation', False)
        self.__progress_file = DictUtils.get(kwargs, 'progress_file', None)
        # Get parameters and variables from a command line/user-provided
        self.params.update(DictUtils.get(kwargs, 'params', {}))
        self.variables.update(DictUtils.get(kwargs, 'vars', {}))

        # Load default configuration
        if not DictUtils.get(kwargs, 'discard_default_config', False):
            logging.debug("Loading default configuration")
            _, self.config, self.param_info = ConfigurationLoader.load(
                os.path.join(os.path.dirname(__file__), 'configs')
            )
        # Load configurations specified on a command line
        self.load_configuration()
        # Add extensions from command line
        DictUtils.ensure_exists(self.config, 'extensions', [])
        self.config['extensions'].extend(DictUtils.get(kwargs, 'extensions', []))
        # All's done
        self.__initialized = True

    def load_configuration(self):
        """Loads configuration specified by a user on a command line.

        At this moment, DLBS has already loaded standard configuration (if `discard_default_config` flag is not
        present). DLBS will try to load user configuration from `config` file (if not None) overwriting default
        parameters. Then, it will try to load user provided parameters (`params`, `vars` and `extensions`) that will
        overwrite existing configuration.

        If `plan` file is present, it will be loaded if `action` is `run`.
        """
        if self.config_file is not None:
            logging.debug('Loading configuration from: %s', self.config_file)
            with open(self.config_file) as file_obj:
                user_config = json.load(file_obj)
                # Update parameter information from user configuration.
                ConfigurationLoader.update_param_info(self.param_info, user_config, is_user_config=True)
                # Update existing benchmark configuration.
                ConfigurationLoader.update(self.config, ConfigurationLoader.remove_info(user_config))
        if self.plan_file is not None and self.action == 'run':
            logging.debug('Loading plan from: %s', self.plan_file)
            with open(self.plan_file) as plan_file:
                self.plan = json.load(plan_file)

    def execute(self):
        """Executed requested action."""
        if self.action == 'print-config':
            json.dump(self.config, sys.stdout, indent=4, sort_keys=True)
            print('')
        elif self.action == 'build':
            self.build_plan(serialize=True)
        elif self.action == 'run':
            self.build_plan()
            logging.info("Plan was built with %d experiments", len(self.plan))
            Processor(self.param_info).compute_variables(self.plan)
            if self.validation:
                validator = Validator(self.plan)
                validator.validate()
                if not validator.plan_ok:
                    validator.report()
                    logging.warn("---------------------------------------------------------------------------")
                    logging.warn("-   Benchmark plan has not been validated. See reason (s) above.          -")
                    logging.warn("-   If you believe validator is wrong (what can very well be the case),   -")
                    logging.warn("-   rerun experimenter with `--no-validation` flag e.g.:                  -")
                    logging.warn("-   python ./python/dlbs/experimenter.py run --no-validation ...          -")
                    logging.warn("---------------------------------------------------------------------------")
                else:
                    logging.info("Benchmark plan has been validated")
            if not self.validation or validator.plan_ok:
                Launcher.run(self.plan, self.__progress_file)
        elif self.action == 'validate':
            self.build_plan()
            Processor(self.param_info).compute_variables(self.plan)
            validator = Validator(self.plan)
            validator.validate()
            validator.report()

    def build_plan(self, serialize=False):
        """Builds plan combining configuration, parameters and variables."""
        self.plan = Builder.build(self.config, self.params, self.variables)
        if serialize:
            if self.plan_file:
                DictUtils.dump_json_to_file(self.plan, self.plan_file)
            else:
                json.dump(self.plan, sys.stdout, indent=4)
                print ('')


def parse_arguments():
    """Parse command line arguments

    Returns:
        dict: Dictionary with command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str,
                        help="Action to perform. Valid actions: 'print-config', 'run', 'build' and 'analyze-plan'.")
    parser.add_argument('--config', required=False, type=str,
                        help="Configuration file (json) of an experiment. Will override values from "
                             "default configuration.")
    parser.add_argument('--plan', required=False, type=str,
                        help="Pre-built plan of an experiment (json file). If action is 'build', a file name to write "
                             "plan to. If action is 'run', a file name to read plan from.")
    parser.add_argument('--progress_file', '--progress-file', required=False, type=str, default=None,
                        help="A JSON file that experimenter will be updating on its progress. "
                             "If not present, no progress info will be available. "
                             "Put it somewhere in /dev/shm")
    parser.add_argument('-P', action='append', required=False, default=[],
                        help="Parameters that override parameters in configuration file. "
                             "For instance, -Pexp.phase=2. Values must be json parsable (json.loads()).")
    parser.add_argument('-V', action='append', required=False, default=[],
                        help="Variables that override variables in configuration file in section 'variables'. "
                             "These variables are used to generate different combinations of experiments. "
                             "For instance: -Vexp.framework='[\"tensorflow\", \"caffe2\"]'. "
                             "Values must be json parsable (json.loads()).")
    parser.add_argument('--log_level', '--log-level', required=False, default='info',
                        help='Python logging level. Valid values: "critical", "error", "warning", "info" and "debug"')
    parser.add_argument('--discard_default_config', '--discard-default-config', required=False, default=False,
                        action='store_true', help='Do not load default configuration.')
    parser.add_argument('--no_validation', '--no-validation', required=False, default=False,
                        action='store_true', help='Do not perform config validation before running benchmarks.')
    parser.add_argument('-E', action='append', required=False, default=[],
                        help="Extensions to add. Can be useful to quickly customize experiments. "
                             "Must be valid json parsable array element for 'extension' array.")
    args = parser.parse_args()
    return vars(args)


def parse_json_arguments(args):
    """Parse parameters, variables and extensions.

    Args:
        args (dict): Dictionary of command line arguments returned by `parse_arguments`. Is not modified.

    Returns:
        A tuple of (params, variables, extensions):
          - `params` is a dictionary of parameters (all params in args['P'])
          - `variables` is a dictionary of variables (all vars in args['V'])
          - `extensions` is a list of dictionaries (all extensions in in args['E'])
    """
    for param in ('P', 'V', 'E'):
        DictUtils.ensure_exists(args, param, [])
    params, variables, extensions = ({}, {}, [])
    DictUtils.add(params, args['P'], pattern='(.+?(?=[=]))=(.+)', must_match=True)
    DictUtils.add(variables, args['V'], pattern='(.+?(?=[=]))=(.+)', must_match=True)
    for extension in args['E']:
        try:
            extensions.append(json.loads(extension))
        except Exception as err:
            logging.warn("Found non-json parsable extension: %s", extension)
            raise err
    return params, variables, extensions


def update_arguments(args, json_args):
    """ Update `args` with data from `json_args`.

    Args:
        args (dict): Dictionary of command line arguments:
          - Keys 'P', 'V' and 'E' are removed.
          - Keys 'params', 'vars' and 'extensions' are created with values from `json_args`.
        json_args (tuple): A tuple returned by `parse_json_arguments` - (params, variables, extensions)

    Returns:
        dict: Updated copy of `args`.
    """
    assert isinstance(json_args, tuple) and len(json_args) == 3, "Invalid type of a function argument"
    args_copy = copy.deepcopy(args)
    for param in ('P', 'V', 'E'):
        del args_copy[param]
    for idx, param in enumerate(['params', 'vars', 'extensions']):
        args_copy[param] = copy.deepcopy(json_args[idx])
    return args_copy


def init_logger(log_level):
    """Initialize logger."""
    if log_level is None:
        return
    log_level = logging.getLevelName(log_level.upper())
    logging.debug("Initializing logger to level %s", log_level)
    root = logging.getLogger()
    root.setLevel(log_level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    root.addHandler(handler)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError("Missing mandatory parameter `action`.")
    if sys.argv[1] == 'help':
        Helper.main()
    elif sys.argv[1] == 'sysinfo':
        print(json.dumps(SysInfo().collect(), indent=2))
    else:
        args = parse_arguments()
        init_logger(args['log_level'])
        args = update_arguments(args, parse_json_arguments(args))
        experimenter = Experimenter()
        experimenter.init(**args)
        experimenter.execute()
