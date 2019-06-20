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
""" This is a system utility that builds several files that then go to docs.

In particular, the script creates the following files:

* ``params.md`` A brief overview of all commonly used parameters. This file is
                part of a documentation file docs/parameters.md
* ``frameworks.md`` Description of parameters that are common to all frameworks.
                    This file is part of docs/frameworks.md
* ``${exp.framework_family}.md`` Several framework-specific files that go into
                                 framework docs in docs/${exp.framework_family}.md.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import json
from dlbs.utils import Six, ConfigurationLoader


class ParamDocBuilder(object):
    """Class that builds parameter documentation page sections."""
    def __init__(self, tmp_folder):
        # Check folder does not exist or empty
        if not os.path.isdir(tmp_folder):
            os.makedirs(tmp_folder)
        else:
            assert len(os.listdir(tmp_folder)) == 0,\
                   "Folder %s must be empty." % tmp_folder
        self.tmp_folder = tmp_folder
        # Load configuration
        _, _, self.param_info = ConfigurationLoader.load(
            os.path.join(os.path.dirname(__file__), '..', 'configs')
        )
        # If a parameter contains description and it is string, convert to array
        for key in self.param_info:
            pi = self.param_info[key]
            if 'desc' in pi and isinstance(pi['desc'], Six.string_types):
                pi['desc'] = [pi['desc']]
        # Load common and framework specific params
        with open(os.path.join(os.path.dirname(__file__), 'frameworks.json')) as file_obj:
            self.framework_info = json.load(file_obj)

    def build(self):
        # Build an overview of parameters
        self.build_param_overview(os.path.join(self.tmp_folder, 'params.md'))
        # Build framework specific param descriptions
        all_params = set(self.param_info.keys())

        def __write_param(fstream, param):
            fstream.write("#### __%s__\n\n" % param)
            param_val = self.param_info[param]['val']
            if isinstance(param_val, Six.string_types):
                f.write("* __default value__ `\"%s\"`\n" % self.param_info[param]['val'])
            else:
                f.write("* __default value__ `%s`\n" % self.param_info[param]['val'])
            f.write("* __description__ " + ' '.join(self.param_info[param]['desc']).replace('(', '\(').replace(')', '\)').replace('"', '\"'))
            f.write('\n\n')

        def __write_framework_params(fstream, common_params, other_params):
            # Write commonly used configuration parameters that are stored in 'common_params'
            fstream.write("## Commonly used configuration parameters\n")
            for param in sorted(common_params):
                __write_param(fstream, param)
            fstream.write("\n")
            fstream.write("## Other parameters\n")
            for param in sorted(other_params):
                __write_param(fstream, param)
            fstream.write("\n")
        # Iterate over framework classes. Each framework class has its own doc file
        # in 'docs' folder.
        for framework_family in ['tensorflow', 'caffe', 'caffe2', 'mxnet', 'tensorrt', 'pytorch']:
            # Get set of framework specific params
            if framework_family == 'caffe':
                frameworks = ('caffe', 'nvidia_caffe', 'bvlc_caffe', 'intel_caffe')
                namespace = ('caffe.', 'nvidia_caffe.', 'bvlc_caffe.', 'intel_caffe.')
                framework_params = {p for p in all_params if p.startswith(namespace)}
                common_params = []
                for framework in frameworks:
                    common_params.extend(self.framework_info[framework])
            else:
                framework_params = {p for p in all_params if p.startswith(framework_family + '.')}
                common_params = self.framework_info[framework_family]
            # Remove these params from set of remaining parameters.
            all_params = all_params - framework_params
            other_params = framework_params - set(common_params)
            # Build a documentation page that consists of common parameters and
            # remaining parameters
            with open(os.path.join(self.tmp_folder, framework_family + '.md'), 'w') as f:
                __write_framework_params(f, common_params, other_params)
        # We need to write common parameters - those that are left in all_params
        with open(os.path.join(self.tmp_folder, 'frameworks.md'), 'w') as f:
            __write_framework_params(
                f,
                self.framework_info['__base__'],
                all_params - set(self.framework_info['__base__'])
            )

    def build_param_overview(self, file_name):
        with open(file_name, 'w') as f:
            def __write_section(title, section_key, ref_file):
                f.write(title + '\n')
                if section_key not in self.framework_info:
                    return
                for param in self.framework_info[section_key]:
                    f.write(
                        "[`%s`](%s?id=%s \"%s\")\n" % (
                            param, ref_file, param.replace('.', ''),
                            ' '.join(self.param_info[param]['desc']).replace('(', '\(').replace(')', '\)').replace('"', '\"')
                        )
                    )
            f.write("## Commonly used parameters\n")
            f.write("\n")
            sections = [
                ("### __Common parameters for all frameworks__", '__base__', '/frameworks/frameworks'),
                ("### __TensorFlow__", 'tensorflow', '/frameworks/tensorflow'),
                ("### __BVLC Caffe__", 'bvlc_caffe', '/frameworks/caffe'),
                ("### __NVIDIA Caffe__", 'nvidia_caffe', '/frameworks/caffe'),
                ("### __Intel Caffe__", 'intel_caffe', '/frameworks/caffe'),
                ("### __Caffe2__", 'caffe2', '/frameworks/caffe2'),
                ("### __MxNet__", 'mxnet', '/frameworks/mxnet'),
                ("### __PyTorch__", 'pytorch', '/frameworks/pytorch'),
                ("### __TensorRT__", 'tensorrt', '/frameworks/tensorrt')
            ]
            for section in sections:
                __write_section(section[0], section[1], section[2])
                f.write("\n")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print ("Usage: %s TMP_FOLDER" % sys.argv[0])
        exit(1)

    builder = ParamDocBuilder(sys.argv[1])
    builder.build()
