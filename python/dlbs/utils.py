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
"""Two classes are define here :py:class:`dlbs.IOUtils` and :py:class:`dlbs.DictUtils`.
"""

import os
import copy
import json
import gzip
import re
import logging
import subprocess
import importlib
from multiprocessing import Process
from multiprocessing import Queue
from glob import glob
from dlbs.exceptions import ConfigurationError


def param2str(val):
    """Convert value val of some benchmark parameter to a string."""
    if isinstance(val, dict):
        try:
            return json.dumps(val)
        except TypeError:
            s = str(val)
            print("[WARNING] cannot convert value ('%s') to a string with json.dumps" % s)

    return str(val)


class OpenFile(object):
    """Class that can work with gzipped and regular textual files."""
    def __init__(self, fname, mode='r'):
        self.__fname = fname
        self.__flags = ['rb', 'r'] if mode == 'r' else ['wb', 'w']

    def __enter__(self):
        if self.__fname.endswith('.gz'):
            self.__fobj = gzip.open(self.__fname, self.__flags[0])
        else:
            self.__fobj = open(self.__fname, self.__flags[1])
        return self.__fobj

    def __exit__(self, type, value, traceback):
        self.__fobj.close()


class IOUtils(object):
    """Container for input/output helpers"""

    @staticmethod
    def mkdirf(file_name):
        """Makes sure that parent folder of this file exists.

        The file itself may not exist. A typical usage is to ensure that we can
        write to this file. If path to parent folder does not exist, it will be
        created.
        See documentation for :py:func:`os.makedirs` for more details.

        :param str file_name: A name of the file for which we want to make sure\
                              its parent directory exists.
        """
        dir_name = os.path.dirname(file_name)
        if dir_name != '' and not os.path.isdir(dir_name):
            os.makedirs(dir_name)

    @staticmethod
    def find_files(directory, file_name_pattern, recursively=False):
        """Find files in a directory, possibly, recursively.

        Find files which names satisfy *file_name_pattern* pattern in folder
        *directory*. If *recursively* is True, scans subfolders as well.

        :param str directory: A directory to search files in.
        :param str file_name_pattern: A file name pattern to search. For instance,
                                      is can be '*.log'
        :param bool recursively: If True, search in subdirectories.
        :return: List of file names satisfying *file_name_pattern* pattern.
        """
        if not recursively:
            files = [f for f in glob(os.path.join(directory, file_name_pattern))]
        else:
            files = [f for p in os.walk(directory) for f in glob(os.path.join(p[0], file_name_pattern))]
        return files

    @staticmethod
    def gather_files(path_specs, file_name_pattern, recursively=False):
        """Find/get files specified by an `inputs` parameter.

        :param list path_specs: A list of file names / directories.
        :param str file_name_pattern: A file name pattern to search. Only
                                      used for entries in path_specs that
                                      are directories.
        :param bool recursively: If True, search in subdirectories. Only used
                                 for entries in path_specs that are directories.
        :return: List of file names satisfying *file_name_pattern* pattern.
        """
        files = []
        for path_spec in path_specs:
            if os.path.isdir(path_spec):
                files.extend(IOUtils.find_files(path_spec, file_name_pattern, recursively))
            elif os.path.isfile(path_spec):
                files.append(path_spec)
        return files

    @staticmethod
    def get_non_existing_file(file_name, max_attempts = 1000):
        """Return file name that does not exist.

        :param str file_name: Input file name.
        :rtype: str
        :return: The 'file_name' if this file does not exist else find first
                 file name that file does not exist.
        """
        if not os.path.exists(file_name):
            return file_name
        attempt = 0
        while True:
            candidate_file_name = "%s.%d" % (file_name, attempt)
            if not os.path.exists(candidate_file_name):
                return candidate_file_name
            attempt += 1
            if attempt >= max_attempts:
                msg = "Cannot find non existing file from pattern %s"
                raise ValueError(msg % file_name)

    @staticmethod
    def check_file_extensions(fname, extensions):
        """Checks that fname has one of the provided extensions.

        :param str fname: The file name to check.
        :param tuple extensions: A tuple of extensions to use.

        Raises exception of fname does not end with one of the extensions.
        """
        if fname is None:
            return
        assert isinstance(extensions, tuple), "The 'extensions' must be a tuple."
        if not fname.endswith(extensions):
            raise ValueError("Invalid file extension (%s). Must be one of %s" % extensions)

    @staticmethod
    def read_json(fname, check_extension=False):
        """Reads JSON object from file 'fname'.

        :param str fname: File name.
        :param boolean check_extension: If True, raises exception if fname does not end
                                        with '.json' or '.json.gz'.
        :rtype: None or JSON object
        :return: None of fname is None else JSON loaded from the file.
        """
        if fname is None:
            return None
        if check_extension:
            IOUtils.check_file_extensions(fname, ('.json', '.json.gz'))
        with OpenFile(fname, 'r') as fobj:
            return json.load(fobj)

    @staticmethod
    def write_json(fname, data, check_extension=False):
        """ Dumps *dictionary* as a json object to a file with *file_name* name.

        :param dict dictionary: Dictionary to serialize.
        :param any data: A data to dump into a JSON file. 
        :param str file_name: Name of a file to serialie dictionary in.
        """
        if fname is None:
            raise ValueError("File name is None")
        if check_extension:
            IOUtils.check_file_extensions(fname, ('.json', '.json.gz'))
        IOUtils.mkdirf(fname)
        with OpenFile(fname, 'w') as fobj:
            json.dump(data, fobj, indent=4)


class DictUtils(object):
    """Container for dictionary helpers."""

    @staticmethod
    def subdict(dictionary, keys):
        """Return subdictionary containing only keys from 'keys'.

        :param dict dictionary: Input dictionary.
        :param list_or_val keys: Keys to extract
        :rtype: dict
        :return: Dictionary that contains key/value pairs for key in keys.
        """
        if keys is None:
            return dictionary
        return dict((k, dictionary[k]) for k in keys if k in dictionary)

    @staticmethod
    def contains(dictionary, keys):
        """Checkes if dictionary contains all keys in 'keys'

        :param dict dictionary: Input dictionary.
        :param list_or_val keys: Keys to find in dictionary
        :rtype: boolean
        :return: True if all keys are in dictionary or keys is None
        """
        if keys is None:
            return True
        keys = keys if isinstance(keys, list) else [keys]
        for key in keys:
            if key not in dictionary:
                return False
        return True

    @staticmethod
    def ensure_exists(dictionary, key, default_value=None):
        """ Ensures that the dictionary *dictionary* contains key *key*

        If key does not exist, it adds a new item with value *default_value*.
        The dictionary is modified in-place.

        :param dict dictionary: Dictionary to check.
        :param str key: A key that must exist.
        :param obj default_value: Default value for key if it does not exist.
        """
        if key not in dictionary:
            dictionary[key] = copy.deepcopy(default_value)

    @staticmethod
    def lists_to_strings(dictionary, separator=' '):
        """ Converts every value in dictionary that is list to strings.

        For every item in *dictionary*, if type of a value is 'list', converts
        this list into a string using separator *separator*.
        The dictictionary is modified in-place.

        :param dict dictionary: Dictionary to modify.
        :param str separator: An item separator.
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

        :param dict dictionary: Dictionary to search keys in.
        :param str prefix: Prefix of keys to be extracted.
        :param bool remove_prefix: If True, remove prefix in returned dictionary.
        :return: New dictionary with items which keys names start with *prefix*.
        """
        return_dictionary = {}
        for key in dictionary:
            if key.startswith(prefix):
                return_key = key[len(prefix):] if remove_prefix else key
                return_dictionary[return_key] = copy.deepcopy(dictionary[key])
        return return_dictionary

    @staticmethod
    def dump_json_to_file(dictionary, file_name):
        """ Dumps *dictionary* as a json object to a file with *file_name* name.

        :param dict dictionary: Dictionary to serialize.
        :param str file_name: Name of a file to serialie dictionary in.
        """
        if file_name is not None:
            IOUtils.mkdirf(file_name)
            with open(file_name, 'w') as file_obj:
                json.dump(dictionary, file_obj, indent=4)

    @staticmethod
    def add(dictionary, iterable, pattern, must_match=True, add_only_keys=None, ignore_errors=False):
        """ Updates *dictionary* with items from *iterable* object.

        This method modifies/updates *dictionary* with items from *iterable*
        object. This object must support ``for something in iterable`` (list,
        opened file etc). Only those items in *iterable* are considered, that match
        *pattern* (it's a regexp epression). If a particular item does not match,
        and *must_match* is True, *ConfigurationError* exception is thrown.

        Regexp pattern must return two groups (1 and 2). First group is considered
        as a key, and second group is considered to be value. Values must be a
        json-parseable strings.

        If *add_only_keys* is not None, only those items are added to *dictionary*,
        that are in this list.

        Existing items in *dictionary* are overwritten with new ones if key already
        exists.

        One use case to use this method is to populate a dictionary with key-values
        from log files.

        :param dict dictionary: Dictionary to update in-place.
        :param obj iterable: Iterable object (list, opened file name etc).
        :param str patter: A regexp pattern for matching items in ``iterable``.
        :param bool must_match: Specifies if every element in *iterable* must match\
                                *pattern*. If True and not match, raises exception.
        :param list add_only_keys: If not None, specifies keys that are added into\
                                   *dictionary*. Others are ignored.
        :param boolean ignore_erros: If true, ignore errors.

        :raises ConfigurationError: If *must_match* is True and not match or if value\
                                    is not a json-parseable string.
        """
        matcher = re.compile(pattern)
        for line in iterable:
            match = matcher.match(line)
            if not match:
                if must_match:
                    raise ConfigurationError("Cannot match key-value from '%s' with pattern '%s'. Must match is set to true" % (line, pattern))
                else:
                    continue
            key = match.group(1).strip()
            try:
                value = match.group(2).strip()
                value = json.loads(value) if len(value) > 0 else None
                if add_only_keys is None or key in add_only_keys:
                    dictionary[key] = value
                    logging.debug("Key-value item (%s=%s) has been parsed and added to dictionary", key, str(value))
            except ValueError as err:
                if not ignore_errors:
                    raise ConfigurationError("Cannot parse JSON string '%s' with key '%s' (key-value definition: '%s'). Error is %s" % (value, key, line, str(err)))

    @staticmethod
    def match(dictionary, query, policy='relaxed', matches=None):
        """ Match *query* against *dictionary*.

        The *query* and *dictionary* are actually dictionaries. If policy is 'strict',
        every key in query must exist in dictionary with the same value to match.
        If policy is 'relaxed', dictionary may not contain all keys from query
        to be matched. In this case, the intersection of keys in dictionary and query
        is used for matching.

        It's assuemd we match primitive types such as numbers and strings not
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

        :param dict dictionary: Dictionary to match.
        :param dict query: Query to use.
        :param ['relaxed', 'strict'] policy: Policy to match.
        :param dict matches: Dictionary where matches will be stored if match has been identified.
        :return: True if match or query is None
        :rtype: bool
        """
        if query is None:
            return True
        assert policy in ['relaxed', 'strict'], ""

        for field, value in query.iteritems():
            if field not in dictionary:
                if policy == 'relaxed':
                    continue
                else:
                    return False
            if isinstance(value, list) or not isinstance(value, basestring):
                values = value if isinstance(value, list) else [value]
                if dictionary[field] not in values:
                    return False
                if matches is not None:
                    matches['%s_0' % (field)] = dictionary[field]
            else:
                if value == '':
                    # Take special care if value is an empty string
                    if value != dictionary[field]:
                        return False
                    elif matches is not None:
                        matches['%s_0' % (field)] = dictionary[field]
                    continue
                else:
                    match = re.compile(value).match(dictionary[field])
                    if not match:
                        return False
                    else:
                        if matches is not None:
                            matches['%s_0' % (field)] = dictionary[field]
                            for index, group in enumerate(match.groups()):
                                matches['%s_%d' % (field, index+1)] = group
                        continue
        return True

class ConfigurationLoader(object):
    """Loads experimenter configuration from multiple files."""

    @staticmethod
    def load(path, files=None):
        """Loads configurations (normally in `conigs`) folder.

        :param str path: Path to load configurations from
        :param list files: List of file names to load. If None, all files with
                           JSON extension in **path** are loaded.
        :return: A tuple consisting of a list of config files, configuration
                 object (dictionary) and dictionary of parameters info

        This method loads configuration files located in 'path'. If `files` is
        empty, all json files are loaded from that folder.
        This method fails if one parameter is defined in multiple files. This
        is intended behaviour for now (this also applies for update_param_info method).
        """
        if path is None:
            raise ValueError("Configuration load error. The 'path' parameter cannot be None.")
        if not os.path.isdir(path):
            raise ValueError("Configuration load error. The 'path' parameter (%s) must point to an existing directory." % path)

        if files is not None:
            config_files = [os.path.join(path, f) for f in files]
        else:
            config_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.json')]
        config = {}         # Configuration with params/vars/extensions
        param_info = {}     # Information on params such as type and help messages
        for config_file in config_files:
            if not os.path.isfile(config_file):
                raise ValueError("Configuration load error. Configuration data cannot be loaded for not a file (%s)" % config_file)
            with open(config_file) as file_obj:
                try:
                    # A part of global configuration from this particular file
                    config_section = json.load(file_obj)
                    # Update parameters info.
                    ConfigurationLoader.update_param_info(param_info, config_section, is_user_config=False)
                    # Joing configuration from this single file.
                    ConfigurationLoader.update(config, ConfigurationLoader.remove_info(config_section))
                except ValueError:
                    logging.error("Configuration load error. Invalid JSON configuration in file %s", config_file)
                    raise
        return (config_files, config, param_info)


    @staticmethod
    def update_param_info(param_info, config, is_user_config=False):
        """Update parameter info dictionary based on configurationi in **config**

        :param dict param_info: A parameter info dictionary that maps parameter
                                name to its description dictionary that contains
                                such fileds as value, help message, type, constraints
                                etc.
        :param dict config: A dictionary with configuration section that may contain
                            parameters, variables and extensions. The **config** is
                            a result of parsing a JSON configuration file.
        :param bool is_user_config: If True, the config object represents user-provided
                                    configuration. If False, this is a system configuration.
                                    Based on this flag, we deal with parameters in config
                                    that redefine parameters in existing param_info
                                    differently. See comments below.

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
                    param_info[name]['val'] = val['val']   # Existing parameter from user configuration, update its value
            else:
                # Just parameter value
                val_type = 'str' if isinstance(val, basestring) or isinstance(val, list) else type(val).__name__
                if name not in param_info:
                    param_info[name] = {
                        'val': val,
                        'type': val_type,
                        'desc': "No description for this parameter provided (it was automatically converted from its value)."
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
        """In parameter section of a **config** the function removes parameter info
        leaving only their values

        :param dict config: A dictionary with configuration section that may contain
                            parameters, variables and extensions. The **config** is
                            a result of parsing a JSON configuration file.
        :return: A copy of **config** with info removed
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
                            "Parameter info remove error."
                            " Parameter that is defined by a dictionary must contain 'val' field that"
                            " defines its default value. Found this definition: %s=%s" % (name, val)
                        )
                    params[name] = val['val']

        return clean_config

    @staticmethod
    def update(dest, source, is_root=True):
        """Merge **source** dictionary into **dest** dictionary assuming source
        and dest are JSON configuration configs or their members.

        :param dict dest: Merge data to this dictionary.
        :param dict source: Merge data from this dictionary.
        :param bool is_root: True if **dest** and *source** are root configuration
                             objects. False if these objects are members.
        """
        def _raise_types_mismatch_config_error(key, dest_val_type, src_val_type, valid_types):
            raise ConfigurationError(
                "Configuration update error - expecting value types to be same and one of %s but"
                " Dest(key=%s, val_type=%s) <- Source(key=%s, val_type=%s)" % (valid_types, key, dest_val_type.__name__, key, src_val_type.__name__)
            )
        # Types and expected key names. Types must always match, else exception is thrown.
        if is_root:
            schema = {'types':(dict, list), 'dict':['parameters', 'variables'], 'list':['extensions']}
        else:
            schema = {'types':(list, basestring, int, float, long)}
        for key in source:
            # Firstly, check that type of value is expected.
            val_type = type(source[key]).__name__
            if not isinstance(source[key], schema['types']):
                raise ConfigurationError(
                    "Configuration update error - unexpected type of key value: "
                    " is_root=%s, key=%s, value type=%s, expected type is one of %s" % \
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
                both_primitive = type(dest[key]) is type(source[key]) and isinstance(dest[key], (basestring, int, float, long))

                if is_root:
                    if not both_dicts and not both_lists:
                        _raise_types_mismatch_config_error(key, type(dest[key]), type(source[key]), '[dict, list]')
                    if both_dicts:
                        ConfigurationLoader.update(dest[key], source[key], is_root=False)
                    else:
                        dest[key].extend(source[key])
                else:
                    if not both_lists and not both_primitive:
                        _raise_types_mismatch_config_error(key, type(dest[key]), type(source[key]), '[list, basestring, int, float, long]')
                    dest[key] = copy.deepcopy(source[key]) if both_lists else source[key]


class ResourceMonitor(object):
    """The class is responsible for launching/shutting down/communicating with
    external resource manager that monitors system resource consumption.

    proc_pid date virt res shrd cpu mem power gpus_power
    """
    def __init__(self, launcher, pid_folder, frequency, fields_specs):
        """Initializes resource monitor but does not create queue and process.

        :param str launcher: A full path to resource monitor script.
        :param str pid_folder: A full path to folder where pid file is created. The
                               file name is fixed and its value is `proc.pid`.
        :param float frequency: A sampling frequency in seconds. Can be something like
                                0.1 seconds
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
            assert len(fields_split) in (3, 4),\
                   "Invalid format of field specification (%s). Must be name:type:index, name:type:index: or name:type:index:count" % raw_field
            field_name = fields_split[0]
            assert field_name not in self.fields,\
                   "Found duplicate timeseries field (%s)" % field_name
            field_type = fields_split[1]
            assert field_type in ('str', 'int', 'float', 'bool'),\
                   "Invalid field type (%s). Must be one of ('str', 'int', 'float', 'bool')" % field_type
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

        :param str launcher: A full path to resource monitor script.
        :param str pid_folder: A full path to folder where pid file is created. The
                               file name is fixed and its value is `proc.pid`.
        :param float frequency: A sampling frequency in seconds. Can be something like
                                0.1 seconds
        :param multiprocessing.Queue queue: A queue to communicate measurements.

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

    @staticmethod
    def str_to_type(str_val, val_type):
        if val_type == 'str':
            return str_val
        elif val_type == 'int':
            return int(str_val)
        elif val_type == 'float':
            return float(str_val)
        elif val_type == 'bool':
            v = str_val.lower()
            assert v in ('true', 'false', '1', '0', 'on', 'off'),\
                   "Invalid boolean value in string (%s)" % str_val
            return v in ('true', 1, 'on')
        else:
            assert False, "Invalid value type %s" % val_type

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
                    metrics[field].append(ResourceMonitor.str_to_type(data[idx], tp))
                elif count == 0:
                    metrics[field].append([ResourceMonitor.str_to_type(data[idx], tp)])
                else:
                    metrics[field].append([
                        ResourceMonitor.str_to_type(data[index], tp) for index in xrange(idx, idx+count)
                    ])
        return metrics

    def remove_pid_file(self):
        """Deletes pif file from disk."""
        try:
            os.remove(self.pid_file)
            p= os.path.dirname(os.path.abspath(self.pid_file))
            os.rmdir(p)
        except OSError:
            logging.error("Error in remove_pid_file {} {}".format(self.pid_file,p))
            pass

    def empty_pid_file(self):
        """Empty pid file."""
        try:
            with open(self.pid_file, 'w'):
                pass
        except IOError:
            pass

    def write_pid_file(self, pid):
        """Write the pid into pid file.

        :param int pid: A pid to write.

        This is a debugging function and most likely should not be used.
        """
        with open(self.pid_file, 'w') as fhandle:
            fhandle.write('%d' % pid)

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

        :param str module_name: A name of a module to try to import, something like
                                'numpy', 'pandas', 'matplotlib' etc.
        :return: True if module has been imported, False otherwise.
        """
        have_module = True
        try:
            importlib.import_module(module_name)
        except ImportError:
            logging.warn("Module '%s' cannot be imported, certain system information will not be available", module_name)
            have_module = False
        return have_module


class Modules(object):
    """A class that enumerates non-standard python modules this project depends on.
    They are optional, so we can disable certain functionality if something is missing.
    """
    HAVE_NUMPY = _ModuleImporter.try_import('numpy')
    HAVE_PANDAS = _ModuleImporter.try_import('pandas')
    HAVE_MATPLOTLIB = _ModuleImporter.try_import('matplotlib')
