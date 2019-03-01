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

# This is a prototype file for logs post-processing functionality. I'll redesign it and add more comments
# shortly. Now I need this functionality as soon as possible, thus it's here in this state.

import sys
import json
from dlbs.utils import DictUtils

# Common model identifiers and their names. Will be moved to some other place.
MODEL_TITLES = {
    "alexnet_owt": "AlexNetOWT",
    "googlenet": "GoogleNet",
    "inception_resnet_v2": "InceptionResNetV2",
    "inception3": "InceptionV3",
    "inception4": "InceptionV4",
    "overfeat": "Overfeat",
    "resnet18": "ResNet18",
    "resnet34": "ResNet34",
    "resnet50": "ResNet50",
    "resnet101": "ResNet101",
    "resnet152": "ResNet152",
    "vgg11": "VGG11",
    "vgg13": "VGG13",
    "vgg16": "VGG16",
    "vgg19": "VGG19",
    "xception": "Xception"
}


class TfCnnBenchmarksBackend(object):

    @staticmethod
    def get_throughput(log_records):
        for log_record in log_records:
            log_record = log_record.strip()
            if log_record.startswith('total images/sec:'):
                return float(log_record[17:].strip())
        return -1

    @staticmethod
    def check(log_records, params, updates):
        # If a newer version is used, performance results are missing.
        throughput = params.get('results.throughput', -1)
        batch_time = params.get('results.time', -1)
        if throughput < 0 or batch_time < 0:
            effective_batch = params.get('exp.effective_batch', -1)
            if effective_batch < 0:
                # This is something completely wrong
                return
            throughput = TfCnnBenchmarksBackend.get_throughput(log_records)
            if throughput > 0:
                updates['results.throughput'] = throughput
                updates['results.time'] = 1000.0 * effective_batch / throughput


def main():
    if len(sys.argv) != 3:
        print("Usage: logger.py BACKEND LOG_FILE")
        exit(1)
    backend = sys.argv[1]
    log_file = sys.argv[2]

    # We may need to iterate multiple times over log records, so, reading log files into
    # a list is a preferable way.
    with open(log_file) as records:
        log_records = [record.strip() for record in records]
    # Parse parameters
    params = {}
    DictUtils.add(params, log_records, pattern='[ \t]*__(.+?(?=__[ \t]*[=]))__[ \t]*=(.+)',
                  must_match=False, ignore_errors=True)
    updates = {}

    # Perform common checks that do not depend on particular backend
    #   1. Check if we need to update a model title
    model = params.get('exp.model', '')
    if model != '' and params.get('exp.model_title', '') == '':
        updates['exp.model_title'] = MODEL_TITLES.get(model, model)

    # Perform checks that depend on a backend type.
    if backend == 'tf_cnn_benchmarks':
        TfCnnBenchmarksBackend.check(log_records, params, updates)

    # Update a log file if needed
    if len(updates) > 0:
        with open(log_file, "a") as file_obj:
            for param in updates:
                file_obj.write("__%s__=%s\n" % (param, json.dumps(updates[param])))


if __name__ == '__main__':
    main()
