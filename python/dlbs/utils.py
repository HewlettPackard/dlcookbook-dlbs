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
"""Multiple classes are define in this module:

  * :py:class:`dlbs.utils.ParamUtils` Various helper methods to work with benchmark parameters.
  * :py:class:`dlbs.utils.OpenFile` Open textual or compressed textual files in one line.
  * :py:class:`dlbs.utils.IOUtils` Various input/output/filesystem helper methods.
  * :py:class:`dlbs.utils.DictUtils` Various methods to work with dictionaries
  * :py:class:`dlbs.utils.ConfigurationLoader` A class that deals with DLBS configuration data.
  * :py:class:`dlbs.utils.ResourceMonitor` A class that implements naive resource monitoring functionality.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import copy
import json
import gzip
import re
import subprocess
import importlib
import logging
from multiprocessing import Process
from multiprocessing import Queue
from glob import glob
from dlbs.exceptions import ConfigurationError


# 99% of the DLBS source code is written in Python2/Python3 compatible mode using only standard modules. Some parts
# however like checking variables type (strings for instance) are Python2/Python3 dependent. Obvious choice is to
# use the `six` library. To run DLBS in a variety of benchmarking environments that may only have standard Python libs,
# this thin `six`-type implementation is used.
class Six(object):
    """A part of functionality found in six library
    https://github.com/benjaminp/six/blob/master/six.py
    """
    PY2 = sys.version_info[0] == 2
    PY3 = sys.version_info[0] == 3

    if PY3:
        string_types = str,
        integer_types = int,
        numeric_types = (int, float)

        @staticmethod
        def iteritems(d, **kw):
            return iter(d.items(**kw))
    else:
        string_types = basestring,
        integer_types = (int, long)
        numeric_types = (int, long, float)

        @staticmethod
        def iteritems(d, **kw):
            return d.iteritems(**kw)


class ParamUtils(object):

    STRING_TYPES = ['str', 'string']
    INTEGER_TYPES = ['int', 'integer']
    NUMERIC_TYPES = ['float', 'numeric']
    BOOLEAN_TYPES = ['bool', 'boolean', 'binary', 'flag']
    TYPES = STRING_TYPES + INTEGER_TYPES + NUMERIC_TYPES + BOOLEAN_TYPES

    BOOLEAN_TRUE_VALUES = ['true', 'on', '1', 'yes']
    BOOLEAN_FALSE_VALUES = ['false', 'off', '0', 'no']
    BOOLEAN_VALUES = BOOLEAN_TRUE_VALUES + BOOLEAN_FALSE_VALUES

    # Pattern that matches ${variable_name} and returns as group(1) variable_name
    VAR_PATTERN = re.compile(r'\$\{([^\}\4\{]+)\}', re.UNICODE)

    @staticmethod
    def to_string(param_value):
        """Convert value `param_value` of some benchmark parameter to a string.

        The function will attempt to converting using JSON methods and if fails, will use `str` function instead.

        Args:
            param_value: A value to convert to a string.

        Returns:
            str: A string representation of `param_value`.

        TODO: Why is only dictionary? What about lists.
        """
        if isinstance(param_value, dict):
            try:
                return json.dumps(param_value)
            except TypeError:
                logging.warning("Cannot convert value ('%s') to a string with json.dumps", str(param_value))
        return str(param_value)

    @staticmethod
    def from_string(value_as_str, value_type):
        """Cast `value_as_str` to a variable of type `value_type`.

        Args:
            value_as_str (str): String representation of a variable stored here.
            value_type (str): Type of a value stored in `value_as_str`.

        Returns:
            A variable of type `value_type` with value parsed from `value_as_str`.
        """
        if value_type in ParamUtils.STRING_TYPES:
            value = value_as_str
        elif value_type in ParamUtils.INTEGER_TYPES:
            value = int(value_as_str)
        elif value_type in ParamUtils.NUMERIC_TYPES:
            value = float(value_as_str)
        elif value_type in ParamUtils.BOOLEAN_TYPES:
            boolean_value = value_as_str.lower()
            if boolean_value not in ParamUtils.BOOLEAN_VALUES:
                raise ValueError("Invalid boolean string (={}). "
                                 "Supported values: {}".format(boolean_value, ParamUtils.BOOLEAN_VALUES))
            value = boolean_value in ParamUtils.BOOLEAN_TRUE_VALUES
        else:
            raise ValueError("Invalid value type (={}). Supported types: {}".format(value_type, ParamUtils.TYPES))
        return value

    @staticmethod
    def is_constant(param_value):
        """ Returns True if `param_value` is a constant value

        Args:
            param_value: A value that must be checked for constness.

        Returns:
            bool: True if `param_value` is constant or False otherwise.
        """
        # If it's not a string, return True
        if not isinstance(param_value, Six.string_types):
            return True
        # Does it reference other parameters? If yes, it's not a constant.
        if ParamUtils.VAR_PATTERN.search(param_value):
            return False
        # Does it contain computable python expression? If yes, it's not a constant.
        idx = param_value.find('$(')
        if idx > 0 and param_value.find(')$', idx+2) > 0:
            return False
        # This is a constant variable.
        return True

    @staticmethod
    def check_value(param_name, param_value, allowed_values=None, regexp=None):
        """ Check if variable contains correct value according to a parameter info.

        Args:
            param_name (str): Name of a parameter.
            param_value (str): Value of a parameter.
            allowed_values (list): List of valid values for this parameter. May be None.
            regexp (str): A regular expression that the value must match. May be None.

        Raises:
            ValueError: If `param_value` violates `allowed_values` or `regexp`.
        """
        # If value is not specified, this is OK.
        if param_value is None:
            return True
        # Value domain check
        if allowed_values is not None and param_value not in allowed_values:
            raise ValueError("Incorrect parameter value found. "
                             "Parameter %s=%s must have value from %s." % (param_name, param_value, allowed_values))
        # Check of regular expression has been provided
        if regexp is not None:
            if re.match(regexp, param_value) is None:
                raise ValueError("Incorrect parameter value found. "
                                 "Parameter %s=%s must match regexp %s." % (param_name, param_value, regexp))

    @staticmethod
    def log_parameters(params, output=sys.stdout):
        """Log parameters in dictionary `params` into an output `output`.

        Args:
            params (dict): Parameters, it is assumed that keys are strings and values are JSON serializable values
            output: An output object to use. Must provide a `write` method accepting string.
        """
        if not isinstance(params, dict):
            raise TypeError("Parameters must be stored in dictionary. But got object of type '%s'" % type(params))
        for param_name in params:
            output.write("__{}__={}\n".format(param_name, json.dumps(params[param_name])))


class OpenFile(object):
    """Class that can work with gzipped and regular textual files.
    Example:
        >>> import json
        >>> my_files = ['~/my_file_1.json', 'my_file_2.jzon.gz']
        >>> for my_file in my_files:
        >>>     with OpenFile(my_file, 'r') as f:
        >>>         print(json.load(f))
    """

    def __init__(self, file_name, mode='r'):
        self.__file_name = file_name
        self.__flags = ['rb', 'r'] if mode == 'r' else ['wb', 'w']
        self.__file_object = None

    def __enter__(self):
        if self.__file_name.endswith('.gz'):
            self.__file_object = gzip.open(self.__file_name, self.__flags[0])
        else:
            self.__file_object = open(self.__file_name, self.__flags[1])
        return self.__file_object

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__file_object.close()


class IOUtils(object):
    """Container for input/output helpers"""

    @staticmethod
    def mkdirf(file_name):
        """Makes sure that parent folder of this file exists.

        The file itself may not exist. A typical usage is to ensure that we can write to this file. If path to parent
        folder does not exist, it will be created. See documentation for :py:func:`os.makedirs` for more details.

        Args:
            file_name (str): A name of the file for which we want to make sure its parent directory exists.
        """
        dir_name = os.path.dirname(file_name)
        if dir_name != '' and not os.path.isdir(dir_name):
            os.makedirs(dir_name)

    @staticmethod
    def find_files(directory, file_name_pattern, recursively=False):
        """Find files in a directory, possibly, recursively.

        Find files which names satisfy *file_name_pattern* pattern in folder `directory`. If `recursively` is True,
        scans sub-folders as well.

        Args:
            directory (str): A directory to search files in.
            file_name_pattern (str): A file name pattern to search. For instance, is can be '*.log'
            recursively (bool): If True, search in subdirectories.

        Returns
            list: List of file names satisfying `file_name_pattern` pattern.
        """
        if not recursively:
            files = [f for f in glob(os.path.join(directory, file_name_pattern))]
        else:
            files = [f for p in os.walk(directory) for f in glob(os.path.join(p[0], file_name_pattern))]
        return files

    @staticmethod
    def gather_files(path_specs, file_name_pattern, recursively=False):
        """Find/get files specified by an `inputs` parameter.

        Args:
            path_specs (list): A list of file names / directories.
            file_name_pattern (str): A file name pattern to search. Only used for entries in path_specs that
                are directories.
            recursively (bool): If True, search in subdirectories. Only used for entries in path_specs that are
                directories.

        Returns:
            list: List of file names satisfying `file_name_pattern` pattern.
        """
        if not isinstance(path_specs, list):
            path_specs = [path_specs]
        files = []
        for path_spec in path_specs:
            if os.path.isdir(path_spec):
                files.extend(IOUtils.find_files(path_spec, file_name_pattern, recursively))
            elif os.path.isfile(path_spec):
                files.append(path_spec)
        return files

    @staticmethod
    def get_non_existing_file(file_name, max_attempts=1000):
        """Return file name that does not exist.

        Args:
            file_name (str): Input file name.
            max_attempts (int): Maximal number of attempts for non-existing files.

        Returns:
            An input file name `file_name` if path does not exist.
            A path in the format `file_name.ATTEMPT` where `ATTEMPT` is the integer between 0 and `max_attempts`.
        """
        if not os.path.exists(file_name):
            return file_name
        for attempt in range(max_attempts):
            candidate_file_name = "{}.{}".format(file_name, attempt)
            if not os.path.exists(candidate_file_name):
                return candidate_file_name
        raise ValueError("Cannot find non existing file from pattern %s" % file_name)

    @staticmethod
    def check_file_extensions(file_name, extensions):
        """Checks that `file_name` has one of the provided extensions.

        Args:
            file_name (str): The file name to check.
            extensions (tuple): A tuple of file extensions to use.

        Raises:
            ValueError: If `file_name` does not end with one of the extensions.
        """
        if file_name is None:
            return
        assert isinstance(extensions, tuple), "The 'extensions' must be a tuple."
        if not file_name.endswith(extensions):
            raise ValueError("Invalid file extension (%s). Must be one of %s" % extensions)

    @staticmethod
    def is_compressed_tarball(input_descriptor):
        """Returns true if this descriptor is a compressed tarball.
        It is an assumption that such files are compressed directories with benchmark log files.
        Args:
            input_descriptor: Input descriptor.
        Returns:
            True if a file name ends with .tgz or .tar.gz
        """
        return isinstance(input_descriptor, Six.string_types) and input_descriptor.endswith(('.tgz', '.tar.gz'))

    @staticmethod
    def is_json_file(input_descriptor):
        """This is quite often used in the project
        Args:
            input_descriptor: Input descriptor.
        Returns:
            True if a file name ends with .json or .json.gz
        """
        return isinstance(input_descriptor, Six.string_types) and input_descriptor.endswith(('.json', '.json.gz'))

    @staticmethod
    def is_csv_file(input_descriptor):
        """Returns true if this is a *.csv file.
        Args:
            input_descriptor: Input descriptor.
        Returns:
            True if a file name ends with .csv or .csv.gz
        """
        return isinstance(input_descriptor, Six.string_types) and input_descriptor.endswith(('.csv', '.csv.gz'))

    @staticmethod
    def read_json(file_name, check_extension=False):
        """Reads JSON object from file 'file_name'.

        Args:
            file_name (str): A file name.
            check_extension (bool): If True, raises exception if fname does not end with '.json' or '.json.gz'.

        Returns:
            None: if `file_name` is None.
            JSON: JSON loaded from the file.
        """
        json_obj = None
        if file_name is not None:
            if check_extension:
                IOUtils.check_file_extensions(file_name, ('.json', '.json.gz'))
            with OpenFile(file_name, 'r') as file_obj:
                json_obj = json.load(file_obj)
        return json_obj

    @staticmethod
    def write_json(output_descriptor, data, check_extension=False):
        """ Dumps `data` as a json object to a file with `file_name` name.
        Args:
            output_descriptor : A specifier of where to write a json object.
            data: A data to dump into a JSON file.
            check_extension (bool): If true, ensure `file_name` has either `json` or `json.gz` extension.
        """
        output_descriptor = output_descriptor if output_descriptor is not None else sys.stdout
        if output_descriptor in [sys.stdout, sys.stderr]:
            json.dump(data, output_descriptor, indent=4)
        elif isinstance(output_descriptor, Six.string_types):
            if check_extension:
                IOUtils.check_file_extensions(output_descriptor, ('.json', '.json.gz'))
            IOUtils.mkdirf(output_descriptor)
            with OpenFile(output_descriptor, 'w') as file_obj:
                json.dump(data, file_obj, indent=4)
        else:
            raise ValueError("Invalid write descriptor ({})".format(type(output_descriptor)))


class DictUtils(object):
    """Container for dictionary helpers."""

    @staticmethod
    def subdict(dictionary, keys):
        """Return sub-dictionary containing only keys from 'keys'.

        Args:
            dictionary (dict): Input dictionary.
            keys: Keys to extract.

        Returns:
            dict: A dictionary that contains key/value pairs for key in `keys`.
        """
        if keys is None:
            return dictionary
        keys = keys if isinstance(keys, list) else [keys]
        return dict((k, dictionary[k]) for k in keys if k in dictionary)

    @staticmethod
    def contains(dictionary, keys):
        """Check if `dictionary` contains all keys in 'keys'.

        Args:
            dictionary (dict): Input dictionary.
            keys: Keys to find in dictionary

        Returns:
            bool: True if all keys are in `dictionary` or `keys` is None.
        """
        if keys is None:
            return True
        keys = keys if isinstance(keys, list) else [keys]
        for key in keys:
            if key not in dictionary:
                return False
        return True

    @staticmethod
    def get(dictionary, key, default_value=None):
        """Return value of `key` in `dictionary` if exists else `default_value`.
        Args:
            dictionary (dict): Dictionary to check.
            key: A key that must exist.
            default_value: Default value for key if it does not exist.
        Returns:
            bool: Value for the `key` if `key` in `dictionary` else `default_value`.
        """
        return dictionary[key] if key in dictionary else default_value

    @staticmethod
    def ensure_exists(dictionary, key, default_value=None):
        """ Ensures that the dictionary *dictionary* contains key *key*

        If key does not exist, it adds a new item with value *default_value*.
        The dictionary is modified in-place.

        Args:
            dictionary (dict): Dictionary to check.
            key: A key that must exist.
            default_value: Default value for key if it does not exist.
        """
        if key not in dictionary:
            dictionary[key] = copy.deepcopy(default_value)

    @staticmethod
    def lists_to_strings(dictionary, separator=' '):
        """ Converts every value in dictionary that is list to strings.

        For every item in `dictionary`, if type of a value is 'list', converts
        this list into a string using separator `separator`.
        The dictionary is modified in-place.

        Args:
            dictionary (dict): Dictionary to modify.
            separator (str): An item separator.
        """
        for key in dictionary:
            if isinstance(dictionary[key], list):
                dictionary[key] = separator.join(str(elem) for elem in dictionary[key])

    @staticmethod
    def filter_by_key_prefix(dictionary, prefix, remove_prefix=True):
        """Creates new dictionary with items which keys start with *prefix*.

        Creates new dictionary with items from *dictionary* which keys
        names starts with *prefix*. If *remove_prefix* is True, keys in new
        dictionary will not contain this prefix.
        The dictionary *dictionary* is not modified.

        Args:
            dictionary (dict): Dictionary to search keys in.
            prefix (str): Prefix of keys to be extracted.
            remove_prefix (bool): If True, remove prefix in returned dictionary.
        Returns:
            dict: New dictionary with items which keys names start with `prefix`.
        """
        return_dictionary = {}
        for key in dictionary:
            if key.startswith(prefix):
                return_key = key[len(prefix):] if remove_prefix else key
                return_dictionary[return_key] = copy.deepcopy(dictionary[key])
        return return_dictionary

    @staticmethod
    def dump_json_to_file(dictionary, file_name):
        """ Dumps `dictionary` as a json object to a file with `file_name` name.
        Args:
            dictionary (dict): Dictionary to serialize.
            file_name (str): Name of a file to serialize dictionary in.
        """
        if file_name is not None:
            IOUtils.mkdirf(file_name)
            with open(file_name, 'w') as file_obj:
                json.dump(dictionary, file_obj, indent=4)

    @staticmethod
    def add(dictionary, iterable, pattern, must_match=True, add_only_keys=None, ignore_errors=False):
        """ Updates `dictionary` with items from `iterable` object.

        This method modifies/updates *dictionary* with items from *iterable*
        object. This object must support ``for something in iterable`` (list,
        opened file etc). Only those items in *iterable* are considered, that match
        *pattern* (it's a regexp expression). If a particular item does not match,
        and *must_match* is True, *ConfigurationError* exception is thrown.

        Regexp pattern must return two groups (1 and 2). First group is considered
        as a key, and second group is considered to be value. Values must be a
        json-parsable strings.

        If *add_only_keys* is not None, only those items are added to *dictionary*,
        that are in this list.

        Existing items in *dictionary* are overwritten with new ones if key already
        exists.

        One use case to use this method is to populate a dictionary with key-values
        from log files.

        Args:
            dictionary (dict): Dictionary to update in-place.
            iterable: Iterable object (list, opened file name etc).
            pattern (str): A regexp pattern for matching items in ``iterable``.
            must_match (bool): Specifies if every element in *iterable* must match `pattern`. If True and not match,
                raises exception.
            add_only_keys (list): If not None, specifies keys that are added into *dictionary*. Others are ignored.
            ignore_errors (bool): If true, ignore errors.

        Raises:
            ConfigurationError: If *must_match* is True and not match or if value is not a json-parsable string.
        """
        matcher = re.compile(pattern)
        for line in iterable:
            match = matcher.match(line)
            if not match:
                if must_match:
                    raise ConfigurationError("Cannot match key-value from '%s' with pattern '%s'. "
                                             "Must match is set to true" % (line, pattern))
                else:
                    continue
            key = match.group(1).strip()
            value = match.group(2).strip()
            try:
                value = json.loads(value) if len(value) > 0 else None
                if add_only_keys is None or key in add_only_keys:
                    dictionary[key] = value
                    logging.debug("Key-value item (%s=%s) has been parsed and added to dictionary", key, str(value))
            except ValueError as err:
                if not ignore_errors:
                    raise ConfigurationError("Cannot parse JSON string '%s' with key '%s' (key-value definition: '%s')."
                                             "Error is %s" % (value, key, line, str(err)))

    @staticmethod
    def match(dictionary, query, policy='relaxed', matches=None):
        """ Match *query* against *dictionary*.

        The *query* and *dictionary* are actually dictionaries. If policy is 'strict',
        every key in query must exist in dictionary with the same value to match.
        If policy is 'relaxed', dictionary may not contain all keys from query
        to be matched. In this case, the intersection of keys in dictionary and query
        is used for matching.

        It's assumed we match primitive types such as numbers and strings not
        lists or dictionaries. If values in query are lists, then condition OR applies.
        For instance:

        match(dictionary, query = { "framework": "tensorflow" }, policy='strict')
           Match dictionary only if it contains key 'framework' with value "tensorflow".
        match(dictionary, query = { "framework": "tensorflow" }, policy='relaxed')
           Match dictionary if it does not contain key 'framework' OR contains\
           key 'framework' with value "tensorflow".
        match(dictionary, query = { "framework": ["tensorflow", "caffe2"] }, policy='strict')
           Match dictionary only if it contains key 'framework' with value "tensorflow" OR\
           "caffe2".
        match(dictionary, query = { "framework": ["tensorflow", "caffe2"], "batch": [16, 32] }, policy='strict')
           Match dictionary only if it (a) contains key 'framework' with value "tensorflow" OR "caffe2"\
           and (b) it contains key 'batch' with value 16 OR 32.

        Args:
            dictionary (dict): Dictionary to match.
            query (dict): Query to use.
            policy (str): Policy to match. One of ['relaxed', 'strict'].
            matches (dict): Dictionary where matches will be stored if match has been identified.
        Returns:
            bool: True if match or query is None.
        """
        if query is None:
            return True
        if isinstance(query, list):
            return DictUtils.contains(dictionary, query)
        assert policy in ['relaxed', 'strict'], ""

        for field, value in Six.iteritems(query):
            if field not in dictionary:
                if policy == 'relaxed':
                    continue
                else:
                    return False
            if isinstance(value, list) or not isinstance(value, Six.string_types):
                values = value if isinstance(value, list) else [value]
                if dictionary[field] not in values:
                    return False
                if matches is not None:
                    matches['%s_0' % field] = dictionary[field]
            else:
                if value == '':
                    # Take special care if value is an empty string
                    if value != dictionary[field]:
                        return False
                    elif matches is not None:
                        matches['%s_0' % field] = dictionary[field]
                    continue
                else:
                    match = re.compile(value).match(dictionary[field])
                    if not match:
                        return False
                    else:
                        if matches is not None:
                            matches['%s_0' % field] = dictionary[field]
                            for index, group in enumerate(match.groups()):
                                matches['%s_%d' % (field, index+1)] = group
                        continue
        return True


class ConfigurationLoader(object):
    """Load benchmark configuration from multiple files."""

    @staticmethod
    def load(path, files=None):
        """Load configurations (normally in dlbs/configs) folder.

        This method loads configuration files located in `path`. If `files` is empty or None, all json files
        are loaded from that folder. This method fails if one parameter is defined in multiple files.
        This is intended behaviour for now (this also applies for update_param_info method).

        Args:
            path (str): Path to load configuration files from.
            files (list): List of file names to load. If None, all files with JSON extension in `path`
                are loaded, else, only those defined in `files` list.

        Returns:
            tuple: A tuple with three elements
              1. A list of config files. This is either all files in `path` or only those files in path that are
                 defined in `files`.
              2. Configuration object. A large dictionary that contains all parameters and extensions found in
                 configuration files listed in config files.
              3. Dictionary of parameter info. Keys are parameters and values are objects defining such parameter
                 information as types and help messages.
        """
        if path is None:
            raise ValueError("Configuration load error. The 'path' parameter cannot be None.")
        if not os.path.isdir(path):
            raise ValueError("Configuration load error. The 'path' parameter (%s) must point "
                             "to an existing directory." % path)

        if files is not None:
            config_files = [os.path.join(path, f) for f in files]
        else:
            config_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.json')]
        config = {}         # Configuration with params/vars/extensions
        param_info = {}     # Information on params such as type and help messages
        for config_file in config_files:
            if not os.path.isfile(config_file):
                raise ValueError("Configuration load error. Configuration data cannot be loaded "
                                 "for not a file (%s)" % config_file)
            with open(config_file) as file_obj:
                try:
                    # A part of global configuration from this particular file
                    config_section = json.load(file_obj)
                    # Update parameters info.
                    ConfigurationLoader.update_param_info(param_info, config_section, is_user_config=False)
                    # Joining configuration from this single file.
                    ConfigurationLoader.update(config, ConfigurationLoader.remove_info(config_section))
                except ValueError:
                    logging.error("Configuration load error. Invalid JSON configuration in file %s", config_file)
                    raise
        return config_files, config, param_info

    @staticmethod
    def update_param_info(param_info, config, is_user_config=False):
        """Update parameter info dictionary based on configuration in `config`

        Args:
            param_info (dict): A parameter info dictionary that maps parameter name to its description
                dictionary that contains such fields as value, help message, type, fieldsaints etc.
            config (dict): A dictionary with configuration section that may contain parameters, variables
                and extensions. The `config` is a result of parsing a JSON configuration file.
            is_user_config (bool): If True, the config object represents user-provided configuration. If False,
                this is a system configuration. Based on this flag, we deal with parameters in config that
                redefine parameters in existing param_info differently. See comments below.

        We are interested here only in parameters section where parameter information
        is defined. There are two scenarios this method is used:
          1. Load standard configuration. In this case, parameter redefinition is
             prohibited. If `parameters` section in `config` redefines existing
             parameters in param_info (already loaded params), program terminates.
          2. Load user-provided configuration. In this case, we still update parameter
             info structure, but deal with it in slightly different way. If parameter in
             `config` exists in param_info, it means user has provided their specific
             value for this parameter.

        Types of user defined parameters are defined either by user in a standard way as
        we define types for standard parameters or induced automatically based on JSON
        parse result.
        """
        if 'parameters' not in config:
            return
        params = config['parameters']
        for name in params:
            val = params[name]
            if not is_user_config:
                # If this is not a user-provided configuration, we disallow parameter redefinition.
                if name in param_info:
                    raise ConfigurationError(
                        "Parameter info update error."
                        " Parameter redefinition is not allowed for non-user configuration."
                        " This is a system configuration error that must not happen."
                        " Parameter %s=%s, new parameter definition (value) is %s" % (name, str(param_info[name]), val)
                    )
            if isinstance(val, dict):
                # This is a complete parameter definition with name, value and description.
                if 'val' not in val:
                    raise ConfigurationError(
                        "Parameter info update error."
                        " Parameter that is defined by a dictionary must contain 'val' field that"
                        " defines its default value. Found this definition: %s=%s" % (name, val)
                    )
                if name not in param_info:
                    param_info[name] = copy.deepcopy(val)  # New parameter, set it info object.
                    # TODO what about parameter type and description?
                else:
                    logging.warn(
                        " Parameter (%s) entirely redefines existing parameter (%s)."
                        " Normally, only value needs to be provided."
                        " We will proceed but you may want to fix this.",
                        json.dumps(val),
                        json.dumps(param_info[name])
                    )
                    param_info[name]['val'] = val['val']  # Existing parameter from user configuration, update its value
            else:
                # Just parameter value
                val_type = 'str' if isinstance(val, Six.string_types) or isinstance(val, list) else type(val).__name__
                if name not in param_info:
                    param_info[name] = {
                        'val': val,
                        'type': val_type,
                        'desc': "No description for this parameter provided "
                                "(it was automatically converted from its value)."
                    }
                else:
                    param_info[name]['val'] = val
            # Do final validations
            if 'type' in param_info[name] and param_info[name]['type'] not in ('int', 'str', 'float', 'bool'):
                raise ConfigurationError(
                    "Parameter info update error."
                    " Parameter has invalid type = '%s'."
                    " Parameter definition is %s = %s" % (param_info[name]['type'], name, param_info[name])
                )
            if 'type' not in param_info[name] or 'desc' not in param_info[name]:
                logging.warn(
                    "Parameter definition does not contain type ('type') and/or description ('desc')."
                    " You should fix this. Parameter definition is"
                    " %s = %s", name, param_info[name]
                )

    @staticmethod
    def remove_info(config):
        """In parameter section of a `config` the function removes parameter info
        leaving only their values

        Args:
            config (dict): A dictionary with configuration section that may contain parameters, variables
                and extensions. The `config` is a result of parsing a JSON configuration file.
        Returns:
            dict: A copy of `config` with info removed. This new dictionary maps parameter name to parameter
                value.
        """
        clean_config = copy.deepcopy(config)

        if 'parameters' in clean_config:
            params = clean_config['parameters']
            for name in params:
                val = params[name]
                if isinstance(val, dict):
                    # This should not generally happen since we deal with it in update_param_info, but just in case
                    if 'val' not in val:
                        raise ConfigurationError(
                            "Parameter info remove error. "
                            "Parameter that is defined by a dictionary must contain 'val' field that "
                            "defines its default value. Found this definition: %s=%s" % (name, val)
                        )
                    params[name] = val['val']

        return clean_config

    @staticmethod
    def update(dest, source, is_root=True):
        """Merge `source` dictionary into `dest` dictionary assuming `source`
        and `dest` are JSON configuration configs or their members.

        Args:
            dest (dict): Merge data to this dictionary.
            source (dict): Merge data from this dictionary.
            is_root (bool): True if `dest` and `source` are root configuration objects. False if these objects are
                members.
        """
        def _raise_types_mismatch_config_error(key, dest_val_type, src_val_type, valid_types):
            raise ConfigurationError(
                "Configuration update error - expecting value types to be same and one of %s but"
                " Dest(key=%s, val_type=%s) <- Source(key=%s, val_type=%s)" % (valid_types, key, dest_val_type.__name__,
                                                                               key, src_val_type.__name__)
            )
        # Types and expected key names. Types must always match, else exception is thrown.
        if is_root:
            schema = {'types': (dict, list), 'dict': ['parameters', 'variables'], 'list': ['extensions']}
        else:
            schema = {'types': (list, float) + Six.string_types + Six.integer_types}
        for key in source:
            # Firstly, check that type of value is expected.
            val_type = type(source[key]).__name__
            if not isinstance(source[key], schema['types']):
                raise ConfigurationError(
                    "Configuration update error - unexpected type of key value: "
                    " is_root=%s, key=%s, value type=%s, expected type is one of %s" %
                    (str(is_root), key, val_type, str(schema['types']))
                )
            # So, the type is expected. Warn if key value is suspicious - we can do it only for root.
            if is_root and key not in schema[val_type]:
                logging.warn("The name of a root key is '%s' but expected is one of '%s'", key, schema[val_type])

            if key not in dest:
                # The key in source dictionary is not in destination dictionary.
                dest[key] = copy.deepcopy(source[key])
            else:
                # The key from source is in dest.
                both_dicts = isinstance(dest[key], dict) and isinstance(source[key], dict)
                both_lists = isinstance(dest[key], list) and isinstance(source[key], list)
                both_primitive = type(dest[key]) is type(source[key]) and \
                                 isinstance(dest[key], Six.string_types + Six.integer_types + (float,))

                if is_root:
                    if not both_dicts and not both_lists:
                        _raise_types_mismatch_config_error(key, type(dest[key]), type(source[key]), '[dict, list]')
                    if both_dicts:
                        ConfigurationLoader.update(dest[key], source[key], is_root=False)
                    else:
                        dest[key].extend(source[key])
                else:
                    if not both_lists and not both_primitive:
                        _raise_types_mismatch_config_error(key, type(dest[key]), type(source[key]),
                                                           '[list, basestring, int, float, long]')
                    dest[key] = copy.deepcopy(source[key]) if both_lists else source[key]


class ResourceMonitor(object):
    """The class is responsible for launching/shutting down/communicating with
    external resource manager that monitors system resource consumption.

    proc_pid date virt res shrd cpu mem power gpus_power
    """
    def __init__(self, launcher, pid_folder, frequency, fields_specs):
        """Initializes resource monitor but does not create queue and process.

        Args:
            launcher (str): A full path to a resource monitor script.
            pid_folder (str): A full path to folder where pid file is created. The file name is fixed and its
                value is `proc.pid`.
            frequency (float): A sampling frequency in seconds. Can be something like 0.1 seconds
            fields_specs (str): A string specifier of what to monitor and how.
        """
        self.launcher = launcher
        self.pid_file = os.path.join(pid_folder, 'proc.pid')
        self.frequency = frequency
        self.queue = None
        self.monitor_process = None
        # Parse fields specs
        # time:str:1,mem_virt:float:2,mem_res:float:3,mem_shrd:float:4,cpu:float:5,mem:float:6,power:float:7,gpus:float:8:
        self.fields = {}
        raw_fields = fields_specs.split(',')
        for raw_field in raw_fields:
            fields_split = raw_field.split(':')
            if len(fields_split) not in (3, 4):
                raise ValueError("Invalid format of field specification (%s). Must be name:type:index, "
                                 "name:type:index: or name:type:index:count" % raw_field)
            field_name = fields_split[0]
            if field_name in self.fields:
                raise ValueError("Found duplicate timeseries field (%s)" % field_name)
            field_type = fields_split[1]
            if field_type not in ('str', 'int', 'float', 'bool'):
                raise ValueError("Invalid field type (%s). Must be one of ('str', 'int', 'float', 'bool')" % field_type)

            index = int(fields_split[2])
            if len(fields_split) == 3:
                count = -1
            elif fields_split[3] == '':
                count = 0
            else:
                count = int(fields_split[3])
            self.fields[field_name] = {
                'type': field_type,
                'index': index,
                'count': count
            }

    @staticmethod
    def monitor_function(launcher, pid_file, frequency, queue):
        """A main monitor worker function.

        Args:
            launcher(str): A full path to resource monitor script.
            pid_file (str): A full path to folder where pid file is created. The file name is fixed and its value is
                `proc.pid`.
            frequency (float): A sampling frequency in seconds. Can be something like 0.1 seconds
            queue (multiprocessing.Queue): A queue to communicate measurements.

        A resource monitor is launched as a subprocess. The thread is reading its
        output and will put the data into a queue. A main thread will then dequeue all
        data at once once experiment is completed.
        """
        cmd = [
            launcher,
            pid_file,
            '',
            str(frequency)
        ]
        process = subprocess.Popen(cmd, universal_newlines=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # The 'output' is a string printed out by a resource monitor
                # script. It's a whitespace separated string of numbers.
                queue.put(output.strip())

    def get_measurements(self):
        """Dequeue all data, put it into lists and return them.
        time:str:1,mem_virt:float:2,mem_res:float:3,mem_shrd:float:4,cpu:float:5,mem:float:6,power:float:7,gpus:float:8-

        :return: Dictionary that maps metric field to a time series of its value.
        """
        metrics = {}
        for key in self.fields.keys():
            metrics[key] = []
        # What's in output:
        #  proc_pid date virt res shrd cpu mem power gpus_power
        while not self.queue.empty():
            data = self.queue.get().strip().split()
            for field in self.fields:
                tp = self.fields[field]['type']
                idx = self.fields[field]['index']
                count = self.fields[field]['count']
                if count == -1:
                    metrics[field].append(ParamUtils.from_string(data[idx], tp))
                elif count == 0:
                    metrics[field].append([ParamUtils.from_string(data[idx], tp)])
                else:
                    metrics[field].append([
                        ParamUtils.from_string(data[index], tp) for index in range(idx, idx+count)
                    ])
        return metrics

    def remove_pid_file(self):
        """Deletes pid file from disk."""
        try:
            os.remove(self.pid_file)
        except OSError:
            pass

    def empty_pid_file(self):
        """Empty pid file."""
        try:
            with open(self.pid_file, 'w'):
                pass
        except IOError:
            pass

    def write_pid_file(self, pid):
        """Write the `pid` (process identifier) into pid file.
        Args:
            pid(int): A pid to write.

        This is a debugging function and most likely should not be used.
        """
        with open(self.pid_file, 'w') as file_obj:
            file_obj.write('%d' % pid)

    def run(self):
        """Create queue and start resource monitor in background thread.

        Due to possible execution of benchmarks in containers, we must not delete
        file here, but create or empty it in host OS.
        """
        self.empty_pid_file()
        self.queue = Queue()
        self.monitor_process = Process(
            target=ResourceMonitor.monitor_function,
            args=(self.launcher, self.pid_file, self.frequency, self.queue)
        )
        self.monitor_process.start()

    def stop(self):
        """Closes queue and waits for resource monitor to finish."""
        with open(self.pid_file, 'w') as fhandle:
            fhandle.write('exit')
        self.queue.close()
        self.queue.join_thread()
        self.monitor_process.join()
        self.remove_pid_file()


class _ModuleImporter(object):
    """A private class that imports a particular models and return boolean
    variable indicating if import has been succesfull or not. Used by a Modules
    class to identify if optional python modules are available.
    """
    @staticmethod
    def try_import(module_name):
        """Tries to import module.
        Args:
            module_name (str): A name of a module to try to import, something like 'numpy', 'pandas', 'matplotlib' etc.
        Returns:
            bool: True if module has been imported, False otherwise.
        """
        have_module = True
        try:
            importlib.import_module(module_name)
        except ImportError:
            logging.warn("Module '%s' cannot be imported, some system information will not be available", module_name)
            have_module = False
        return have_module


class Modules(object):
    """A class that enumerates non-standard python modules this project depends on.
    They are optional, so we can disable certain functionality if something is missing.
    """
    HAVE_NUMPY = _ModuleImporter.try_import('numpy')
    HAVE_PANDAS = _ModuleImporter.try_import('pandas')
    HAVE_MATPLOTLIB = _ModuleImporter.try_import('matplotlib')
