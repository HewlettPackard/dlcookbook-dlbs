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
"""The processor (template engine) given dictionary of variables expands them.

Sort of reference implementation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging
# Do not remove these imports. It is used in configuration files to generate experiments UUIDs and in other cases.
import uuid  # noqa # pylint: disable=unused-import
import sys   # noqa # pylint: disable=unused-import
import re    # noqa # pylint: disable=unused-import
#
import json
from dlbs.utils import Six
from dlbs.exceptions import ConfigurationError
from dlbs.exceptions import LogicError
from dlbs.utils import DictUtils
from dlbs.utils import ParamUtils


class Processor(object):
    """Class that computes variables in all experiments.

    Another terms for this is variable expansion or template substitution or
    template engine.
    """

    def __init__(self, param_info=None):
        """ Constructor

        Args:
            param_info (dict): A dictionary that maps parameter name to its description. This description should
                include default value, type, help message and optional constraints such as value domain.

        Forward index (self.fwd_index) is an updatable dictionary that maps variable name to the following object:
            {
                'deps': set(),      # All variables this variable depends on. Does not change once computed
                                    # and if `finalized` is True.
                'udeps': set()      # Unsatisfied variable dependencies up to current moment. Updatable field.
                'finalized': True   # True if `deps` list needs not to be recomputed. It's False for nested
                                    # variables - which names themselves are defined by other variables, for
                                    # instance, ${${exp.fork}_caffe.host.libpath}.
            }
        """
        self.param_info = param_info
        self.fwd_index = {}

    def report_unsatisfied_deps(self, experiment):
        """Report to a user unsatisfied dependencies for variables.

        Args:
            experiment (dict): Experiment for which variables computations failed.
        """
        undefined_vars = set()
        defined_vars = {}
        print("====================== ERROR ======================")
        print("Missing dependencies. Variables cannot be computed.")
        print("===================================================")
        print("Forward index:")
        for var in self.fwd_index:
            print("\t%s: %s" % (var, str(self.fwd_index[var]['udeps'])))
            for udep_var in self.fwd_index[var]['udeps']:
                if udep_var not in experiment:
                    undefined_vars.add(udep_var)
                else:
                    defined_vars[udep_var] = experiment[udep_var]
        print("===================================================")
        print("Undefined variables:")
        print("\t%s" % (str(undefined_vars)))
        print("If this set is empty, this indicates a bug in this program (class Processor in processor.py).")
        print("Else, you need to define this variables. We do not allow for undefined variables to have empty values.")
        print("===================================================")
        print("Defined variables:")

        print(json.dumps(defined_vars, sort_keys=False, indent=4))
        print("===================================================")

    def compute_variables(self, experiments):
        """Main entry point - compute all variables in all experiments.

        Args:
            experiments (list): A list of experiments that needs to be computed. It's modified in place.
        """
        for experiment in experiments:
            # Convert all lists to strings
            DictUtils.lists_to_strings(experiment)
            # Build initial version of a forward index (variables -> their dependencies)
            self.fwd_index = {}
            for variable in experiment:
                self.update_index(experiment, variable)
            # iteratively compute variables
            while len(self.fwd_index) > 0:
                computable_vars = self.get_computable_variables()
                # print("Computable vars: %s" % (str(computable_vars)))
                if len(computable_vars) == 0:
                    self.report_unsatisfied_deps(experiment)
                    exit(1)
                # Compute  variables. We are either done with a variable or
                # this variable has nested references and we need to continue
                # computing it.
                computed, partially_computed = self.compute_current_variables(experiment, computable_vars)
                # print("Computed vars: %s" % (str(computed)))
                # print("Partially computed vars: %s" % (str(partially_computed)))
                # Remove computed vars from index and update dependencies of
                # remaining variables
                for computed_var in computed:
                    self.fwd_index.pop(computed_var)
                for var in self.fwd_index:
                    self.fwd_index[var]['udeps'].difference_update(set(computed))
                # Update partially computed variables - these are variables
                # that have nested references.
                for var in partially_computed:
                    self.update_index(experiment, var)
                    deps = self.fwd_index[var]['udeps'].copy()
                    for dep in deps:
                        if dep not in self.fwd_index:
                            self.fwd_index[var]['udeps'].remove(dep)
                # exit(0)

            # We need to remove all internal temp variables
            # We need to remove all internal temp variables.
            # In P2, keys() makes a copy. In P3 it returns an iterator -> this
            # 'dictionary changed size during iteration' error. So, making copy
            for name in list(experiment.keys()):
                if name.startswith('__dlbs_'):
                    experiment.pop(name)

    def update_index(self, experiment, variable):
        """Updates forward index for given `variable`.

        Args:
            experiment (dict): Current experiment.
            variable (str): Variable name.
        """
        if variable not in self.fwd_index:
            # Finalized means there are no nested variables
            # deps - dependencies, udeps - unsatisfied dependencies
            self.fwd_index[variable] = {'deps': set(), 'udeps': set(), 'finalized': True}
        if isinstance(experiment[variable], Six.string_types):
            # Add inner most variables
            found_variables = 0
            for match in ParamUtils.VAR_PATTERN.finditer(experiment[variable]):
                # print(match.group(1))
                dep = match.group(1)
                self.fwd_index[variable]['deps'].add(dep)
                # Add to unsatisfied deps only if this variable is in experiment
                # or is undefined
                if dep in experiment or dep not in os.environ:
                    self.fwd_index[variable]['udeps'].add(dep)
                found_variables += 1
            assert variable not in self.fwd_index[variable]['deps'], \
                "Cyclic dependency found for %s=%s" % (variable, experiment[variable])
            num_opening_tags = experiment[variable].count('${')
            assert not (found_variables == 0 and num_opening_tags > 0), \
                "Number of opening tags '${' is %d and no variables found in '%s'" % (num_opening_tags, experiment[variable])
            self.fwd_index[variable]['finalized'] = (num_opening_tags == found_variables)

    def get_computable_variables(self):
        """ Return variables that can be computed at current step.

        These are variables that do not depend on other variables (*udeps* set is
        empty) or those which dependencies are environmental variables.

        Returns:
            list: List of variable names that can be computed.
        """
        computable_vars = []
        for var in self.fwd_index:
            # Check of var does not depend on other variables
            if len(self.fwd_index[var]['udeps']) == 0:
                computable_vars.append(var)
                continue
            # Check if all dependencies in os.environ
            init_from_env = True
            for var_dep in self.fwd_index[var]['udeps']:
                if var_dep not in os.environ:
                    init_from_env = False
                    break
            if init_from_env:
                computable_vars.append(var)
        return computable_vars

    def compute_current_variables(self, experiment, computable_variables):
        """Computes all variables in `experiment` that are in `computable_variables`.

        Args:
            experiment (dict): Current experiment.
            computable_variables (list): Names of variables that need to be computed.

        Returns:
            tuple: (computed (list), partially_computed(list))

        The computed variables are those that have been computed and their values can be used. The partially computed
        variables are those that contain nested references (`finalized` is set to False for them).
        """
        computed = []
        partially_computed = []
        for var in computable_variables:
            is_str = isinstance(experiment[var], Six.string_types)
            if not is_str:
                computed.append(var)
                continue

            if is_str and len(self.fwd_index[var]['deps']) > 0:
                for ref_var in self.fwd_index[var]['deps']:
                    replace_pattern = "${%s}" % ref_var
                    if ref_var in experiment:
                        replace_value = ParamUtils.to_string(experiment[ref_var])
                    elif ref_var in os.environ:
                        replace_value = ParamUtils.to_string(os.environ[ref_var])
                    else:
                        msg = "Cannot determine value of the parameter %s = %s because variable `%s` not found. "\
                              "Either this variable is not in the list of benchmark parameters, or this may happen "\
                              "if variable's name depends on other variable that's empty or set to an incorrect "\
                              "value. For instance, the name of ${${exp.framework}.docker.image} variable depends "\
                              "on ${exp.framework} value. If it's empty, the variable name becomes '.docker.image' "\
                              "what's wrong."
                        raise LogicError(msg % (var, experiment[var], ref_var))
                    experiment[var] = experiment[var].replace(replace_pattern, replace_value)

            # Search for computable components
            while True:
                idx = experiment[var].find('$(')
                if idx < 0:
                    break
                end_idx = experiment[var].find(')$', idx+2)
                if end_idx < 0:
                    raise ConfigurationError("Cannot find ')$' in %s. Variable cannot be computed" % (experiment[var]))
                try:
                    eval_res = eval(experiment[var][idx+2:end_idx])
                except NameError as err:
                    logging.error("Cannot evaluate python expression: %s", experiment[var][idx+2:end_idx])
                    raise err
                logging.debug("\"%s\" -> \"%s\"", experiment[var][idx+2:end_idx], str(eval_res))
                experiment[var] = experiment[var][:idx] + str(eval_res) + experiment[var][end_idx+2:]

            if self.fwd_index[var]['finalized'] is True:
                computed.append(var)
                self.cast_variable(experiment, var)
                self.check_variable_value(experiment, var)
            else:
                partially_computed.append(var)

        return computed, partially_computed

    def cast_variable(self, experiment, var):
        """Cast variable `var` defined in `experiment` to its true type.

        The cast operation is only defined for variables that are 'string' variables by default. The reason why we
        want to have this op is because in JSON configs we can define variables that depend on other variables and/or
        that are python computable expressions. The result is always a string, so we need to be able to cast it to an
        appropriate type to be able to use such variables in a standard way to define other variables.

        Args:
            experiment (dict): A dictionary with experiment parameters. Will be modified in-place.
            var (str): A name of a parameter from this experiment.

        """
        # Type of this parameter must be string:
        if not isinstance(experiment[var], Six.string_types):
            return
        # Parameter info dictionary must present and contain info on this parameter
        if self.param_info is None or var not in self.param_info:
            return
        # An information object must contain type info:
        if 'type' not in self.param_info[var]:
            return
        #
        experiment[var] = ParamUtils.from_string(experiment[var], self.param_info[var]['type'])

    def check_variable_value(self, experiment, var):
        """ Check if variable contains correct value according to parameter info.

        Args:
            experiment (dict): A dictionary with experiment parameters.
            var (str): Name of a parameter.
        """
        if self.param_info is None or var not in self.param_info:
            return
        pi = self.param_info[var]
        ParamUtils.check_value(var,                                     # Parameter name
                               experiment[var],                         # Parameter value
                               DictUtils.get(pi, 'val_domain', None),   # Value domain constraints.
                               DictUtils.get(pi, 'val_regexp', None))   # Value regexp constraints.
