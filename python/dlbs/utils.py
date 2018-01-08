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
import re
import logging
import subprocess
from multiprocessing import Process
from multiprocessing import Queue
from glob import glob
from dlbs.exceptions import ConfigurationError


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
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

    @staticmethod
    def find_files(directory, file_name_pattern, recursively=False):
        """Find files in a directory, possibly, recursively.

        Find files which names satisfy *file_name_pattern* pattern in folder
        *directory*. If *recursively* is True, scans subfolders as well.

        :param str directory: A directory to search files in.
        :param str file_name_pattern: A file name pattern to search.
        :param bool recursively: If True, search in subdirectories.
        :return: List of file names satisfying *file_name_pattern* pattern.
        """
        if not recursively:
            files = [f for f in glob(os.path.join(directory, file_name_pattern))]
        else:
            files = [f for p in os.walk(directory) for f in glob(os.path.join(p[0], file_name_pattern))]
        return files


class DictUtils(object):
    """Container for dictionary helpers."""

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
    def add(dictionary, iterable, pattern, must_match=True, add_only_keys=None):
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
            except ValueError:
                raise ConfigurationError("Cannot parse JSON string '%s' with key '%s' (key-value definition: '%s')" % (value, key, line))
            if add_only_keys is None or key in add_only_keys:
                dictionary[key] = value
                logging.debug("Key-value item (%s=%s) has been parsed and added to dictionary", key, str(value))

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
        :return: True if match
        :rtype: bool
        """
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
    def load(path):
        """Loads configurations (normally in `conigs`) folder.

        :param str path: Path to load configurations from
        :return: A tuple consisting of a list of config files and configuration 
                 object (dictionary)
        """
        config_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.json')]
        config = {}
        for config_file in config_files:
            logging.debug('Loading configuration from: %s', config_file)
            with open(config_file) as file_obj:
                try:
                    ConfigurationLoader.update(config, json.load(file_obj))
                except ValueError as error:
                    logging.error("Invalid JSON configuration in file %s", config_file)
                    raise error
        return config_files, config

    @staticmethod
    def update(obj1, obj2, is_root=True):
        """Merge **obj2** into **obj1** assuming obj1 and obj2 are JSON
        configuration configs or their members.

        :param dict obj1: Merge data to this dictionary.
        :param dict obj2: Merge data from this dictionary.
        :param bool is_root: True if **obj1** and *obj2** are root configuration
                             objects. False if these objects are members.
        """
        for key in obj2:
            if key not in obj1:
                obj1[key] = copy.deepcopy(obj2[key])
            else:
                both_dicts = isinstance(obj1[key], dict) and isinstance(obj2[key], dict)
                both_lists = isinstance(obj1[key], list) and isinstance(obj2[key], list)
                both_primitive = type(obj1[key]) is type(obj2[key]) and isinstance(obj1[key], (basestring, int, float, long))

                if is_root:
                    assert both_dicts or both_lists, "In root configuration objects, only dictionaries and lists are allowed."
                    if both_dicts:
                        # This is for parameters and variables section
                        ConfigurationLoader.update(obj1[key], obj2[key], is_root=False)
                    else:
                        # This works for extensions
                        obj1[key].extend(obj2[key])
                else:
                    assert both_lists or both_primitive, "Members of configuration must be either lists or primitive types"
                    obj1[key] = copy.deepcopy(obj2[key]) if both_lists else obj2[key]


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
