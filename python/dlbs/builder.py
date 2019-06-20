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
"""Builder builds experiments.

Experiment is just a bunch of named parameters. Builder takes as an input standard
configurations, configuration, provided by a user, parameters and variables passed on
a command line and using cartesian product of variables plus extensions builds as many
experiments as possible.

Variables in experiments are not computed.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import copy
import uuid
from dlbs.utils import Six, DictUtils, ParamUtils


class Builder(object):
    """Builds experiments' plans but does not compute their variables."""

    @staticmethod
    def build(config, params, variables):
        """ Given input configuration and command line parameters/variables build experiments

        Args:
            config (dict): Dictionary of parameters/variables/extensions
            params (dict): Dictionary of command line parameters
            variables (dict): Dictionary of command line variables

        Returns:
            list: Array of experiments. Each experiment is defined by a set of parameters.

        A high level overview of what builder does is:
        ::

          1. Add **variables** to 'variables' section of a configuration ('config').
          2. Override variables in 'parameters' section in 'config' with those specified on a command line ('params').
          3. For every combination (Cartesian product) of variables in 'config':
             a. Create copy of parameters.
             b. Add combination to those parameters
             c. Apply extensions possibly generating multiple experiments
             d. Add them to a list of experiments.

        In case input configuration contains extensions, this algorithm applies:
        ::

          1. Set experiments <- [experiment]
          2. For extension in extension_list:
          3.     Set active_experiments <- []
          4.     For experiment in experiments:
          5.         If not match(experiment, extension.condition):
          6.             active_experiments.append(experiment)
          7.         active_experiments.extend( extend(extended_config, extension) )
          8.     experiments <- active_experiments
          9. Return experiments
        """
        # This makes parsing life easier.
        for section in ['parameters', 'variables', 'extensions']:
            DictUtils.ensure_exists(config, section, {})
        for extension in config['extensions']:
            for section in ['condition', 'parameters', 'cases']:
                DictUtils.ensure_exists(extension, section, {})
        # First, we need to update variables that contribute to creating
        # different experiments
        for var_name in variables:
            config['variables'][var_name] = copy.deepcopy(variables[var_name])
        # We also need to ensure that those values are lists
        for variable in config['variables']:
            if not isinstance(config['variables'][variable], list):
                config['variables'][variable] = [config['variables'][variable]]

        # Now, we need to override environmental variables
        for param in params:
            config['parameters'][param] = copy.deepcopy(params[param])

        plan = []
        # Get order of variables in experiments
        # These are all variables that we will vary
        var_keys = config['variables'].keys()
        # Filter oder of those variables - remove ones that we do not actually have
        if 'sys.plan_builder.var_order' in config['parameters']:
            var_order = [k for k in config['parameters']['sys.plan_builder.var_order'] if k in var_keys]
        else:
            var_order = []
        # Add those that are not in an order array
        for k in var_keys:
            if k not in var_order:
                var_order.append(k)

        var_values = [config['variables'][var_key] for var_key in var_order]
        # This loop will work just once if var_values is empty.
        for variables_combination in itertools.product(*var_values):
            # Create base set of variables.
            experiment = copy.deepcopy(config['parameters'])
            # Add current combination of variables
            experiment.update(dict((k, v) for (k, v) in zip(var_order, variables_combination)))
            # Apply extensions possibly generating many experiment configurations
            extended_experiments = Builder.apply_extensions(experiment, config)
            # Add to plan
            plan.extend(extended_experiments)
        return plan

    @staticmethod
    def apply_extensions(base_experiment, config):
        """ Apply extensions in `config` to experiment `base_experiment`.

        The algorithm looks like this. We start with a list containing only
        one experiment - `base_experiment`. Then, we each extension we try to
        extend all experiments in a list.

        Args:
            base_experiment (dict): Parameters of an experiment
            config (dict): Configuration dictionary

        Returns:
            list: List of experiments extended with extensions or list with `base_experiment`.
        """
        experiments = [copy.deepcopy(base_experiment)]
        for extension in config['extensions']:
            # in 'base_experiment' dictionary.
            active_experiments = []
            for experiment in experiments:
                session_id = uuid.uuid4().__str__().replace('-', '')
                # Condition matches will indicate what was matched in the form "field_%d: value"
                # where %d is an integer number. 0 indicates entire match, other
                # indicates groups if present.

                # Now, condition may only be used when referenced parameter in
                # 'condition' section is a constant (does not depend on other parameters)
                Builder.assert_match_is_corrent(experiment, extension['condition'])

                matches = {}
                if not DictUtils.match(experiment, extension['condition'], policy='relaxed', matches=matches):
                    # Not a match, keep unmodified version of this experiment
                    active_experiments.append(copy.deepcopy(experiment))
                else:
                    # Create base extended version using 'parameters' section
                    # of an extension
                    extension_experiment = copy.deepcopy(experiment)
                    # Add condition matched variables in case they are referenced by parameters or cases
                    for match_key in matches:
                        session_key = '__dlbs_%s_%s' % (session_id, match_key)
                        extension_experiment[session_key] = matches[match_key]
                    # We need to update values in `extension["parameters"]` for
                    # current session id
                    extension_experiment.update(Builder.correct_var_ref_in_extension(session_id,
                                                                                     extension['parameters']))
                    if len(extension['cases']) == 0:
                        active_experiments.append(extension_experiment)
                    else:
                        for case in extension['cases']:
                            case_experiment = copy.deepcopy(extension_experiment)
                            # We need to update values in `case` for current session id
                            case_experiment.update(Builder.correct_var_ref_in_extension(session_id, case))
                            active_experiments.append(case_experiment)

            experiments = active_experiments

        experiments = [experiment for experiment in experiments if len(experiment) > 0]
        return experiments

    @staticmethod
    def assert_match_is_corrent(experiment, condition):
        """ Checks that parameters in `condition` have constant values in `experiment`.

        Args:
            experiment (dict): Dictionary of parameters for current experiment
            condition (dict): Dictionary of parameters constraints. Here, we are interested only in parameter names

        If match cannot be performed correctly, program terminates. Incorrect match is a match when parameter in
        `experiment` is not constant i.e. depends on other parameters.
        """
        for param in condition:
            # If parameter not in experiment, just do not consider it
            if param not in experiment:
                continue
            if not ParamUtils.is_constant(experiment[param]):
                raise ValueError("Condition must not use parameter that's not a constant (%s=%s)" %
                                 (param, experiment[param]))

    @staticmethod
    def correct_var_ref_in_extension(session_id, params):
        """Correct variables references in `params` and 'cases' that reference extension variables.

        Args:
            session_id (str): Unique identifier of an extension for this experiment.
            params (str): Values in this dictionary must be scanned for correction. This dictionary is not modified
                here.

        Returns:
            dict: Dictionary with corrected variable references.
        """
        session_prefix = '${__dlbs_%s_' % session_id
        new_params = copy.deepcopy(params)
        for key, value in new_params.items():
            # assert not isinstance(value, list), "Lists are not supported in extension section"
            if isinstance(value, Six.string_types):
                new_params[key] = value.replace('${__condition.', session_prefix)
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    if isinstance(item, Six.string_types):
                        value[idx] = item.replace('${__condition.', session_prefix)
        return new_params
