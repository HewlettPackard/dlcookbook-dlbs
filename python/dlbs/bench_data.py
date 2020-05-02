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

"""
LogParser:
    $ python bench_data.py parse inputs --recursive --output [FILENAME]
BenchStats:
    $ python bench_data.py summary [FILENAME] --select SELECT --update UPDATE
SummaryBuilder
    $ python bench_data.py report  [FILENAME] --select SELECT --update UPDATE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import json
import csv
import argparse
import itertools
import os
import sys
import pickle
import tarfile
from dlbs.utils import Six, IOUtils, OpenFile, DictUtils
from dlbs.processor import Processor


class DLPGUtils(object):

    # Parameters that are expected to be of string type.
    STRING_PARAMS = ["exp.proj_name", "exp.proj_parent_name", "exp.proj_description", "exp.experimenter_name",
                     "exp.experiment_name", "exp.experiment_description", "exp.framework_title", "exp.framework_ver",
                     "exp.backend", "exp.dlbs_hashtag", "exp.software", "exp.node_id", "exp.node_title", "exp.node_nic",
                     "exp.device_type", "exp.device_title", "exp.id", "exp.dtype", "exp.data", "exp.phase",
                     "exp.model_title"]
    # Parameters that are expected to be of numeric (integers/floating point numbers) type.
    NUMERIC_PARAMS = ["results.throughput", "results.time"]
    # Parameters that are expected to be of integer type.
    INTEGER_PARAMS = ["exp.num_node_gpus", "exp.num_nodes", "exp.effective_batch", "exp.replica_batch",
                      "exp.num_local_replicas"]
    # Some commonly used values, failing these tests does not necessarily mean that compliance tests have failed.
    EXPECTED_VALUES = {
        "exp.framework_title": ["TensorFlow", "Caffe", "Caffe2", "MXNET", "PyTorch", "TensorRT"],
        "exp.backend": ["caffe", "caffe2", "mxnet", "nvcnn", "nvtfcnn", "pytorch", "tensorflow", "tensorrt"],
        "exp.node_id": ["apollo_6500_xl_gen9", "apollo_6500_xl_gen10"],
        "exp.node_title": ["Apollo 6500 XL Gen9", "Apollo 6500 XL Gen10"],
        "exp.device_type": ["cpu", "gpu"],
        "exp.device_title": ["Tesla P100-PCIE-16GB", "Tesla P100-SXM2-16GB",
                             "Tesla P4", "Tesla T4",
                             "Tesla V100-PCIE-16GB", "Tesla V100-SXM2-16GB", "Tesla V100-SXM2-32GB"],
        "exp.num_node_gpus": [1, 2, 4, 8],
        "exp.dtype": ["float32", "float16", "int8"],
        "exp.data": ["synthetic", "real", "real/ssd", "real/dram", "real/weka", "real/nvme"],
        "exp.phase": ["training", "inference"],
        "exp.model_title": ["AcousticModel", "AlexNet", "AlexNetOWT", "DeepMNIST", "DeepSpeech2", "GoogleNet",
                            "InceptionV3", "InceptionV4", "Overfeat", "ResNet18", "ResNet34", "ResNet50",
                            "ResNet101", "ResNet152", "ResNet200", "ResNet269", "SensorNet", "Seq2SeqAutoencoder",
                            "TextCNN", "VGG11", "VGG13", "VGG16", "VGG19"]
    }

    @staticmethod
    def report_check_status(test_failed, test_name, not_ok_message):
        if test_failed:
            print("[FAILED]  [{}]  '{}'".format(test_name, not_ok_message))
        else:
            print("[OK]      [{}]  'All tests passed.'".format(test_name))

    @staticmethod
    def check_missing_parameters(bench_data):
        params = DLPGUtils.STRING_PARAMS + DLPGUtils.NUMERIC_PARAMS + DLPGUtils.INTEGER_PARAMS
        failed_parameters = set()
        for benchmark in bench_data.benchmarks():
            failed_parameters.update(par for par in params if par not in benchmark)
        DLPGUtils.report_check_status(failed_parameters, "Missing parameters      ",
                                      "Not found: {}".format(failed_parameters))

    @staticmethod
    def check_parameter_type(bench_data, params, types, human_type):
        failed_parameters = set()
        for benchmark in bench_data.benchmarks():
            failed_parameters.update(par for par in params if not isinstance(benchmark.get(par, None), types))
        DLPGUtils.report_check_status(failed_parameters, "Parameter type ({})".format(human_type),
                                      "TypeOf({}) must be one of ({})".format(failed_parameters,
                                                                              [t.__name__ for t in types]))

    @staticmethod
    def check_values_positive(bench_data):
        params = DLPGUtils.NUMERIC_PARAMS + DLPGUtils.INTEGER_PARAMS
        failed_parameters = set()
        for benchmark in bench_data.benchmarks():
            failed_parameters.update(par for par in params if benchmark[par] <= 0)
        DLPGUtils.report_check_status(failed_parameters, "Parameters > 0 test     ",
                                      "These parameters must be positive (>0): {}".format(failed_parameters))

    @staticmethod
    def check_values_not_empty(bench_data):
        params = set(DLPGUtils.STRING_PARAMS) - set(('exp.dlbs_hashtag', 'exp.proj_parent_name', 'exp.node_nic'))
        failed_parameters = set()
        for benchmark in bench_data.benchmarks():
            failed_parameters.update(par for par in params if benchmark.get(par, None) == "")
        DLPGUtils.report_check_status(failed_parameters, "Parameters == '' test   ",
                                      "These parameters must not be empty (!= ''): {}".format(failed_parameters))

    @staticmethod
    def check_values(param, param_values, expected_values):
        unexpected_values = [val for val in param_values if val not in expected_values]
        DLPGUtils.report_check_status(unexpected_values, "Parameter values        ",
                                      "{} = {} but should be one of : {}".format(param, unexpected_values,
                                                                                 expected_values))

    @staticmethod
    def check(bench_data):
        print("==== DLPG Compliance Tests ====")

        DLPGUtils.check_missing_parameters(bench_data)
        DLPGUtils.check_parameter_type(bench_data, DLPGUtils.STRING_PARAMS, Six.string_types, ' string')
        DLPGUtils.check_parameter_type(bench_data, DLPGUtils.NUMERIC_PARAMS, Six.numeric_types, 'numeric')
        DLPGUtils.check_parameter_type(bench_data, DLPGUtils.INTEGER_PARAMS, Six.integer_types, 'integer')
        DLPGUtils.check_values_positive(bench_data)
        DLPGUtils.check_values_not_empty(bench_data)

        summary = bench_data.summary(params=list(DLPGUtils.EXPECTED_VALUES))
        for param in DLPGUtils.EXPECTED_VALUES:
            DLPGUtils.check_values(param, summary[param], DLPGUtils.EXPECTED_VALUES[param])

        print("===============================")


def print_vals(obj):
    """A helper to print JSON with predefined indent. Is widely used in python notebooks.

    Args:
        obj: Something to print with json.dumps.
    """
    print(json.dumps(obj, indent=2))


class BenchData(object):

    @staticmethod
    def get_selector(query):
        """Returns a callable object that returns true when `query` matches dictionary.

        Args:
            query: An object that specifies the query. It can be one of the following:
                - A string:
                    - Load JSON object from this string if possible ELSE
                    - Treat it as a file name and load JSON objects from there.
                  The parsed/loaded object must be either dict or list.
                - A list of dict. Wrap it into a function that calls match method of a DictUtils class.
                - Callable object. Return as is.

        Returns:
            Callable object.
        """
        # If it's a string, assume it's a JSON parsable string and if not - assume it's a JSON file name.
        if isinstance(query, Six.string_types):
            try:
                query = json.loads(query)
            except ValueError:
                query = IOUtils.read_json(query)

        selector = query
        # If it's a list of dict, wrap it into a function.
        if isinstance(query, (list, dict)):
            def dict_matcher(bench): return DictUtils.match(bench, query, policy='strict')
            selector = dict_matcher
        # Here, it must be a callable object.
        if not callable(selector):
            raise ValueError("Invalid type of object that holds parameters (%s)" % type(selector))
        return selector

    @staticmethod
    def status(arg):
        """ Return status of the benchmark stored in a log file `log_file`.

        Args:
            arg: A name of a log file, a dictionary or an instance of the BenchData class.

        Returns:
            str or None: "ok" for successful benchmark, "failure" for not and None for other cases (such as no file).
        """
        if isinstance(arg, Six.string_types):
            bench_data = BenchData.parse(arg)
        elif isinstance(arg, dict):
            bench_data = BenchData([arg], create_copy=False)
        elif isinstance(arg, BenchData):
            bench_data = arg
        else:
            raise TypeError("Invalid argument type (={}). Expecting string, BenchData".format(type(arg)))
        if len(bench_data) == 1:
            return 'ok' if DictUtils.get(bench_data[0], 'results.time', -1) > 0 else 'failure'
        return None

    @staticmethod
    def load(inputs, **kwargs):
        """Load benchmark data (parsed from log files) from a JSON file.

        A file name is a JSON file that contains object with 'data' field. This field
        is a list with dictionaries, each dictionary contains parameters for one benchmark:
        {"data":[{...}, {...}, {...}]}

        Args:
            inputs (str): File name of a JSON (*.json) or a compressed JSON (.json.gz) file.

        Returns:
            Instance of this class.
        """
        is_json_file = IOUtils.is_json_file(inputs)
        if not is_json_file and isinstance(inputs, list) and len(inputs) == 1:
            is_json_file = IOUtils.is_json_file(inputs[0])
            inputs = inputs[0] if is_json_file else inputs
        if is_json_file:
            benchmarks = IOUtils.read_json(inputs, check_extension=True)
            if 'data' not in benchmarks:
                benchmarks = {'data': []}
                print("[WARNING]: No benchmark data found in '{}'".format(inputs))
            return BenchData(benchmarks['data'], create_copy=False)
        #
        is_csv_file = IOUtils.is_csv_file(inputs)
        if not is_csv_file and isinstance(inputs, list) and len(inputs) == 1:
            is_csv_file = IOUtils.is_csv_file(inputs[0])
            inputs = inputs[0] if is_csv_file else inputs
        if is_csv_file:
            with OpenFile(inputs, 'r') as fobj:
                reader = csv.DictReader(fobj)
                benchmarks = list(reader)
            return BenchData(benchmarks, create_copy=False)
        #
        is_compressed_tarball = IOUtils.is_compressed_tarball(inputs)
        if not is_compressed_tarball and isinstance(inputs, list) and len(inputs) == 1:
            is_compressed_tarball = IOUtils.is_json_file(inputs[0])
            inputs = inputs[0] if is_compressed_tarball else inputs
        if is_compressed_tarball:
            benchmarks = []
            with tarfile.open(inputs, "r:gz") as archive:
                for member in archive.getmembers():
                    if member.isfile() and member.name.endswith('.log'):
                        log_file = archive.extractfile(member)
                        if log_file is not None:
                            parameters = {}
                            DictUtils.add(
                                parameters,
                                log_file,
                                pattern='[ \t]*__(.+?(?=__[ \t]*[=]))__[ \t]*=(.+)',
                                must_match=False,
                                ignore_errors=True
                            )
                            benchmarks.append(parameters)
            return BenchData(benchmarks, create_copy=False)
        #
        return BenchData.parse(inputs, **kwargs)

    @staticmethod
    def parse(inputs, recursive=False, ignore_errors=False):
        """Parse benchmark log files (*.log).

        Args:
            inputs: Path specifiers of where to search for log files.
            recursive (bool): If true, parse directories found in `inputs` recursively.
            ignore_errors (bool): If true, ignore errors associated with parsing parameter values.

        Returns:
            Instance of this class.
        """
        inputs = inputs if isinstance(inputs, list) else [inputs]
        log_files = set()
        for file_path in inputs:
            if os.path.isdir(file_path):
                log_files.update(IOUtils.gather_files(inputs, "*.log", recursive))
            elif file_path.endswith('.log'):
                log_files.add(file_path)
        log_files = list(log_files)
        benchmarks = []
        for log_file in log_files:
            parameters = {}
            with OpenFile(log_file, 'r') as logfile:
                # The 'must_match' must be set to false. It says that not
                # every line in a log file must match key-value pattern.
                DictUtils.add(
                    parameters,
                    logfile,
                    pattern='[ \t]*__(.+?(?=__[ \t]*[=]))__[ \t]*=(.+)',
                    must_match=False,
                    ignore_errors=ignore_errors
                )
            benchmarks.append(parameters)
        return BenchData(benchmarks, create_copy=False)

    @staticmethod
    def merge_benchmarks(dest_dict, source_dict):
        for bench_key in source_dict:
            if bench_key not in dest_dict:
                dest_dict[bench_key] = source_dict[bench_key]
            else:
                dest_dict[bench_key].update(source_dict[bench_key])

    def __init__(self, benchmarks=None, create_copy=False):
        if benchmarks is None:
            self.__benchmarks = []
        else:
            self.__benchmarks = copy.deepcopy(benchmarks) if create_copy else benchmarks

    def __len__(self):
        """Return number of benchmarks.

        Returns:
            Number of benchmarks.
        """
        return len(self.__benchmarks)

    def __getitem__(self, i):
        """Return parameters for the i-th benchmark

        Args:
            i (int): Benchmark index.

        Returns:
            dict: Parameters for i-th benchmark.
        """
        return self.__benchmarks[i]

    def as_dict(self, key_len=-1):
        """ Return dictionary mapping benchmark id to its parameters
        Args:
            key_len (int): Length of a benchmark key. By default(-1) entire value of a `exp.id` (which is GUID)
                will be used.
        Returns:
            dict: Mapping from benchmark id to their parameters
        """
        benchmarks = {}
        for benchmark in self.__benchmarks:
            if 'exp.id' not in benchmark:
                continue
            key = benchmark['exp.id']
            if key_len > 0:
                bench_key_len = min(key_len, len(benchmark['exp.id']))
                key = benchmark['exp.id'][0:bench_key_len]
            benchmarks[key] = benchmark
        return benchmarks

    def benchmarks(self):
        """Return list of dictionaries with benchmark parameters.

        Returns:
            List of dictionaries where each dictionary contains parameters for one benchmarks.
        """
        return self.__benchmarks

    def clear(self):
        """Remove all benchmarks."""
        self.__benchmarks = []

    def copy(self):
        """Create a copy of this bench data instance.

        Returns:
            Copy of this instance.
        """
        return BenchData(copy.deepcopy(self.__benchmarks))

    def save(self, output_descriptor):
        """ Save contents of this instance into a (compressed) JSON file.

        Args:
            output_descriptor (str): A file name.
        """
        IOUtils.write_json(output_descriptor, {'data': self.__benchmarks})

    def select(self, query):
        """ Select only those benchmarks that match `query`

        Args:
            query: Anything that's a valid with respect to `get_selector` method.

        Returns:
            BenchData: That contains those benchmarks that match `query`.
        """
        match = BenchData.get_selector(query)
        selected = [bench for bench in self.__benchmarks if match(bench)]
        return BenchData(selected, create_copy=False)

    def delete(self, query):
        """ Delete only those benchmarks that match `query`

        Args:
            query: Anything that's a valid with respect to `get_selector` method.

        Returns:
            BenchData: That contains those benchmarks that do not match `query`.
        """
        match = BenchData.get_selector(query)
        return self.select(lambda bench: not match(bench))

    def update(self, query, use_processor=False):
        """Update benchmarks returning updated copy.

        Args:
            query: dict or callable.
            use_processor (bool): If true, apply variable processor. Will silently produce wrong results if
                benchmarks contain values that are dicts or lists.

        Returns:
            BenchData: Updated copy of benchmarks.
        """
        update_fn = query
        if isinstance(query, dict):
            def dict_update_fn(bench): bench.update(query)
            update_fn = dict_update_fn
        if not callable(update_fn):
            raise ValueError("Invalid update object (type='%s'). Expecting callable." % type(update_fn))

        benchmarks = copy.deepcopy(self.__benchmarks)
        for benchmark in benchmarks:
            update_fn(benchmark)

        if use_processor:
            Processor().compute_variables(benchmarks)
        return BenchData(benchmarks, create_copy=False)

    def select_keys(self, keys):
        """Return copy of benchmarks that only contain `keys`

        Args:
            keys (list): List of benchmark keys to keep

        Returns:
            BenchData: Copy of current benchmarks with parameters defined in `keys`.
        """
        if keys is None:
            return self.copy()
        selected = [copy.deepcopy(DictUtils.subdict(bench, keys)) for bench in self.__benchmarks]
        return BenchData(selected, create_copy=False)

    def select_values(self, key):
        """Return unique values for the `key` across all benchmarks.

        A missing key in a benchmark is considered to be a key having None value.

        Args:
            key (str): A key to return unique values for.

        Returns:
            list: sorted list of values.
        """
        selected = set()
        for benchmark in self.__benchmarks:
            selected.add(DictUtils.get(benchmark, key, None))
        return sorted(list(selected))

    def summary(self, params=None):
        """Return summary of benchmarks providing additional info on `params`.

        Args:
            params (list): List of parameters to provide additional info for. If empty, default list is used.

        Returns:
            dict: A summary of benchmarks.
        """
        if not params:
            params = ['exp.node_id', 'exp.node_title', 'exp.gpu_title', 'exp.gpu_id', 'exp.framework_title',
                      'exp.framework_id']
        summary_dict = {
            'num_benchmarks': len(self.__benchmarks),
            'num_failed_benchmarks': 0,
            'num_successful_benchmarks': 0
        }
        for param in params:
            summary_dict[param] = set()

        for bench in self.__benchmarks:
            if DictUtils.get(bench, 'results.time', -1) > 0:
                summary_dict['num_successful_benchmarks'] += 1
            else:
                summary_dict['num_failed_benchmarks'] += 1
            for param in params:
                summary_dict[param].add(DictUtils.get(bench, param, None))

        for param in params:
            summary_dict[param] = list(summary_dict[param])
        return summary_dict

    def report(self, inputs=None, output=None, output_cols=None,
               report_speedup=False, report_efficiency=False, **kwargs):
        """
        Args:
            inputs (list): List of "input" columns that identify a one report record key. Each report record must
                have unique key.
            output (str): An output column name (like exp.replica_batch).
            output_cols (list): If given, titles for output columns. Number of output columns is equal to number of distinct
                values of the output parameter.
            report_speedup (bool): If true, output speedup table.
            report_efficiency (bool): If true, output efficiency table.
        """
        reporter = BenchData.Reporter(self)
        reporter.report(inputs, output, output_cols, report_speedup, report_efficiency, **kwargs)

    class Reporter(object):
        TITLES = {
            'exp.model_title': 'Model', 'exp.replica_batch': 'Replica Batch', 'exp.effective_batch': 'Effective Batch',
            'exp.num_gpus': 'Num GPUs', 'exp.gpus': 'GPUs', 'exp.dtype': 'Precision',
            'exp.docker_image': 'Docker Image', 'exp.device_type': 'DeviceType'
        }

        @staticmethod
        def to_string(val):
            if val is None:
                return "-"
            elif isinstance(val, Six.string_types):
                return val
            elif isinstance(val, Six.integer_types):
                return "{:d}".format(val)
            elif isinstance(val, float):
                return "{:.2f}".format(val)
            else:
                raise TypeError("Invalid value type (='{}'). Expecting strings, integers or floats.".format(type(val)))

        def build_cache(self, inputs=None, output=None, output_cols=None):
            self.input_cols = [None] * len(inputs)
            for idx, param in enumerate(inputs):
                self.input_cols[idx] = {"index": idx, "param": param, "width": 0,
                                        "title": DictUtils.get(BenchData.Reporter.TITLES, param, param),
                                        "vals": sorted(self.bench_data.select_values(param))}
            self.output_param = output
            output_cols = output_cols if output_cols else sorted(self.bench_data.select_values(output))
            self.output_cols = [None] * len(output_cols)
            for idx, param_value in enumerate(output_cols):
                self.output_cols[idx] = {"index": idx, "value": param_value, "title": param_value,
                                         "width": len(BenchData.Reporter.to_string(param_value))}
            self.cache = {}
            print("Number of benchmarks = {}".format(len(self.bench_data.benchmarks())), file=sys.stderr)
            for bench in self.bench_data.benchmarks():
                if BenchData.status(bench) != "ok":
                    print("Ignoring failed benchmark: exp.id={}, exp.status={}, results.time={}, exp.model={}, "
                          "exp.replica_batch={}, exp.dtype={}, exp.num_gpus={}.".format(
                            bench.get('exp.id', 'UNKNOWN'), bench.get('exp.status', 'UNKNOWN'),
                            bench.get('results.time', -1), bench.get('exp.model', 'UNKNOWN'),
                            bench.get('exp.replica_batch', 'UNKNOWN'), bench.get('exp.dtype', 'UNKNOWN'),
                            bench.get('exp.num_gpus', -1)), file=sys.stderr)
                    continue
                # The 'bench_key' is the composite benchmark ID that includes values of input and output variable, for
                # instance ['VGG16', 128, 2] may mean [ModelTitle, ReplicaBatch, NumGPUs].
                bench_key = []
                # Build initial version of the key taking into account input parameters.
                for input_col in self.input_cols:
                    # The param_value is the value of an output parameter, for instance, number of GPUs
                    param_value = DictUtils.get(bench, input_col['param'], None)
                    if not param_value:
                        bench_key = []
                        break
                    bench_key.append(str(param_value))
                if bench_key:
                    output_val = DictUtils.get(bench, self.output_param, None)
                    if output_val:
                        bench_key = '.'.join(bench_key + [str(output_val)])
                        if bench_key not in self.cache:
                            self.cache[bench_key] = bench
                        else:
                            raise ValueError("Duplicate benchmark with key = {}".format(bench_key))
                    else:
                        pass

        def compute_column_widths(self, times, throughputs):
            # Input columns
            for input_col in self.input_cols:
                input_col['width'] = len(input_col['title'])
                for val in input_col['vals']:
                    input_col['width'] = max(input_col['width'], len(BenchData.Reporter.to_string(val)))
            # Output columns
            num_rows = len(times)
            num_output_cols = len(self.output_cols)
            for row_idx in range(num_rows):
                for col_idx in range(num_output_cols):
                    self.output_cols[col_idx]['width'] = max([
                        self.output_cols[col_idx]['width'],
                        len(BenchData.Reporter.to_string(times[row_idx][col_idx])),
                        len(BenchData.Reporter.to_string(throughputs[row_idx][col_idx]))
                    ])

        def compute_speedups(self, throughputs):
            speedups = copy.deepcopy(throughputs)
            num_cols = len(self.output_cols)
            for row in speedups:
                for idx in range(1, num_cols):
                    row[idx] = None if row[0] is None or row[idx] is None else float(row[idx]) / row[0]
                row[0] = 1.00 if row[0] is not None else None
            return speedups

        def compute_efficiency(self, times):
            replica_batch_idx = -1
            effective_batch_idx = -1
            for col_idx, input_col in enumerate(self.input_cols):
                if input_col['param'] == "exp.replica_batch":
                    replica_batch_idx = col_idx
                elif input_col['param'] == "exp.effective_batch":
                    effective_batch_idx = col_idx

            if (replica_batch_idx == -1 and effective_batch_idx == -1) or \
               (replica_batch_idx >= 0 and effective_batch_idx >= 0) or \
               self.output_param != "exp.num_gpus":
                raise ValueError("Efficiency can only be computed when one of the inputs is either replica or "
                                 "effective batch and when output is the number of GPUs e.g: "
                                 "inputs=['exp.model_title', 'exp.replica_batch'], output='exp.num_gpus'")
            efficiency = copy.deepcopy(times)
            num_cols = len(self.output_cols)
            for row in efficiency:
                for idx in range(1, num_cols):
                    if row[0] is None or row[idx] is None:
                        row[idx] = None
                    else:
                        if replica_batch_idx >= 0:
                            # Weak scaling
                            row[idx] = int(10000.0 * row[0] / row[idx]) / 100.0
                        else:
                            # String scaling
                            row[idx] = int(10000.0 * row[0] / (self.output_cols[idx]['value'] * row[idx])) / 100.0
                        row[idx] = min(row[idx], 100.0)
                row[0] = 100.00 if row[0] is not None else None
            return efficiency

        def get_header(self):
            header = ""
            for input_col in self.input_cols:
                format_str = "  %-" + str(input_col['width']) + "s"
                header = header + format_str % BenchData.Reporter.to_string(input_col['title'])
            header += "    "
            output_cols_title = " " * len(header) + DictUtils.get(BenchData.Reporter.TITLES,
                                                                  self.output_param, self.output_param)
            for output_col in self.output_cols:
                format_str = "%+" + str(output_col['width']) + "s  "
                header = header + format_str % BenchData.Reporter.to_string(output_col['title'])
            return [output_cols_title, header]

        def print_table(self, title, header, inputs, outputs):
            print(title)
            for header_line in header:
                print(header_line)
            for input, output in zip(inputs, outputs):
                row = ""
                for input_col in self.input_cols:
                    format_str = "  %-" + str(input_col['width']) + "s"
                    row = row + format_str % BenchData.Reporter.to_string(input[input_col['index']])
                row += "    "
                num_missing_outputs = 0
                for output_col in self.output_cols:
                    format_str = "%+" + str(output_col['width']) + "s  "
                    val = BenchData.Reporter.to_string(output[output_col['index']])
                    if val == '-':
                        num_missing_outputs += 1
                    row = row + format_str % val
                if num_missing_outputs != len(self.output_cols):
                    print(row)
            print("\n\n")

        def __init__(self, bench_data):
            self.bench_data = bench_data
            self.input_cols = None
            self.output_param = None
            self.output_cols = None
            self.cache = None

        def report(self, inputs=None, output=None, output_cols=None,
                   report_speedup=False, report_efficiency=False, **kwargs):
            DictUtils.ensure_exists(kwargs, 'report_batch_times', True)
            DictUtils.ensure_exists(kwargs, 'report_input_specs', True)
            # Build cache that will map benchmarks keys to benchmark objects.
            self.build_cache(inputs, output, output_cols)
            # Iterate over column values and build table with batch times and throughput
            cols = []
            times = []
            throughputs = []
            benchmark_keys = [input_col['vals'] for input_col in self.input_cols]
            # Build tables for batch times and benchmarks throughputs
            # The `benchmark_key` is a tuple of column values e.g. ('ResNet50', 256)
            for benchmark_key in itertools.product(*benchmark_keys):
                cols.append(copy.deepcopy(benchmark_key))
                times.append([None] * len(self.output_cols))
                throughputs.append([None] * len(self.output_cols))
                for output_col in self.output_cols:
                    benchmark_key = [str(key) for key in benchmark_key]
                    key = '.'.join(benchmark_key + [str(output_col['value'])])
                    if key in self.cache:
                        times[-1][output_col['index']] = self.cache[key]['results.time']
                        throughputs[-1][output_col['index']] = self.cache[key]['results.throughput']
                    else:
                        pass
            # Determine minimal widths for columns
            self.compute_column_widths(times, throughputs)
            #
            header = self.get_header()
            if kwargs['report_batch_times']:
                self.print_table("Batch time (milliseconds)", header, cols, times)
            self.print_table("Throughput (instances per second e.g. images/sec)", header, cols, throughputs)
            if report_speedup:
                speedups = self.compute_speedups(throughputs)
                self.print_table("Speedup (based on instances per second table, "
                                 "relative to first output column ({} = {}))".format(self.output_param,
                                                                                     self.output_cols[0]['value']),
                                 header, cols, speedups)
            if report_efficiency:
                efficiency = self.compute_efficiency(times)
                self.print_table("Efficiency (based on batch times table, "
                                 "relative to first output column ({} = {}))".format(self.output_param,
                                                                                     self.output_cols[0]['value']),
                                 header, cols, efficiency)
            if kwargs['report_input_specs']:
                print("This report is configured with the following parameters:")
                print(" inputs = %s" % str(inputs))
                print(" output = %s" % output)
                print(" output_cols = %s" % str(output_cols))
                print(" report_speedup = %s" % str(report_speedup))
                print(" report_efficiency = %s" % str(report_efficiency))


def parse_arguments():
    """Parse command line arguments

    Returns:
        dict: Dictionary with command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str, default='parse', choices=['parse', 'summary', 'report', 'benchdb'],
                        help="Action to perform. ")

    parser.add_argument('inputs', type=str, nargs='*',
                        help="Input file(s) and/or folders. ")
    parser.add_argument('--no-recursive', '--no_recursive', required=False, default=False,
                        action='store_true', help='When parsing log files, do not parse folders recursively.')
    parser.add_argument('--ignore_errors', required=False, default=False, action='store_true',
                        help="If set, ignore errors related to parsing benchmark parameters.")

    parser.add_argument('--select', type=str, required=False, default=None,
                        help="A select query to filter benchmarks.")
    parser.add_argument('--update', type=str, required=False, default=None,
                        help="An expression to update query benchmarks.")
    parser.add_argument('--output', type=str, required=False, default=None,
                        help="File to write output to. If not specified, standard output is used. When log parsing "
                             "is performed, several output formats are supported: '*.json' and '*.json.gz'.")

    parser.add_argument('--report', type=str, required=False, default=None,
                        help="A type of report to build - one of (regular|weak|strong) or a JSON parsable string. "
                             "It must be a dictionary with such keys as inputs, output and optionally report_speedup "
                             "and report_efficiency.")
    return vars(parser.parse_args())


class BenchDataApp(object):
    REPORT_PARAMS = {
        'regular': {
            'inputs': ['exp.model_title', 'exp.device_type'], 'output': 'exp.replica_batch'
        },
        'weak': {
            'inputs': ['exp.model_title', 'exp.replica_batch'], 'output': 'exp.num_gpus',
            'report_speedup': True, 'report_efficiency': True
        },
        'strong': {
            'inputs': ['exp.model_title', 'exp.effective_batch'], 'output': 'exp.num_gpus',
            'report_speedup': True, 'report_efficiency': True
        }
    }

    def __init__(self, args):
        self.__args = args
        self.__data = None
        self.__functions = {
            'parse': self.action_parse,
            'summary': self.action_summary,
            'report': self.action_report,
            'benchdb': self.action_benchdb
        }

    def load(self):
        data = BenchData.load(self.__args['inputs'],
                              recursive=not self.__args['no_recursive'],
                              ignore_errors=self.__args['ignore_errors'])
        if self.__args['select'] is not None:
            data = data.select(self.__args['select'])
        if self.__args['update'] is not None:
            data = data.update(self.__args['select'], use_processor=False)
        return data

    def action_parse(self):
        print("Number of benchmarks: {}".format(len(self.__data)), file=sys.stderr)
        self.__data.save(self.__args['output'])

    def action_summary(self):
        IOUtils.write_json(self.__args['output'], self.__data.summary(), check_extension=False)

    def action_report(self):
        if self.__args['report'] in BenchDataApp.REPORT_PARAMS:
            report_params = BenchDataApp.REPORT_PARAMS[self.__args['report']]
        else:
            try:
                report_params = json.loads(self.__args['report'])
            except ValueError:
                print("Invalid format of a report specification: {}".format(self.__args['report']))
                exit(1)
        self.__data.report(**report_params)

    def action_benchdb(self):
        #
        print("Searching for benchmark archives ...")
        file_names = []
        for file_type in ('*.tgz', '*.tar.gz', '*.json.gz'):
            file_names.extend(IOUtils.find_files(self.__args['inputs'][0], file_type, recursively=True))
        print("    found {} benchmark files.".format(len(file_names)))
        #
        bench_data = {}
        print("Parsing benchmark archives ...")
        for file_name in file_names:
            BenchData.merge_benchmarks(
                bench_data,
                BenchData.load(file_name).as_dict(key_len=5)
            )
            print("    done [{}]".format(file_name))

        print("    found {} benchmarks.".format(len(bench_data)))
        #
        print("Serializing benchmarks ...")
        with open('/dev/shm/benchmark_db.pickle', 'wb') as handle:
            pickle.dump(bench_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        print("    database generation completed.")

    def run(self):
        action = self.__args['action']
        if action != 'benchdb':
            self.__data = self.load()
        if action in self.__functions:
            self.__functions[action]()


if __name__ == "__main__":
    # This module does not work with Python3 if input / output files are *.gz files. Use Python2 instead, or
    # use raw json without compression. Google for this error:
    #   "json.dump gzip typeerror memoryview a bytes-like object is required"
    # To fix it, possible changes are required in DLPG.
    app = BenchDataApp(parse_arguments())
    app.run()
