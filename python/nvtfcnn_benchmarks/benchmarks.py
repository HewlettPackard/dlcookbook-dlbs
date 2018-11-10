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
"""nvtfcnn benchmarks entry point module.
   Runs the Nvidia NGC Docker image Horovod CNN benchmarks.
"""
from __future__ import absolute_import
from __future__ import print_function
import argparse
import pathlib
import sys
import re

from subprocess import call
models_re=['alexnet$','inception_resnet_v2$','inception_v4$',r'resnet_\d+$',r'vgg_\d+$',
            'googlenet$','inception_v3$','overfeat$','trivial$','xception$']
def allowed_models(v):
    v=v.strip()
    for model_re in models_re:
        if re.match(model_re,v): break
    else:
        raise argparse.ArgumentTypeError('The specified model {} does not match any recognized pattern ({})'.format(v,','.join(models_re)))
    return v

def main():
    """Main worker function."""
    parser = argparse.ArgumentParser(add_help=False)
    #Update here if list of available models change.
    parser.add_argument( '--model', type=allowed_models, required=True,
                         default='', 
                         help="A model to benchmark - must match one of the patterns: {}".format(' | '.join(models_re)))
    args, passthru = parser.parse_known_args()
    m=re.match("(resnet|vgg)_(\d+)",args.model)
    if m:
        args.model=m.group(1)
        passthru+=["--layers {}".format(m.group(2))]

    prog=pathlib.Path(__file__).parent.joinpath('cnn').joinpath(args.model+".py")
    cmd=['python',str(prog)] + passthru
    try:
        retcode = call(cmd)
        if retcode < 0:
            print("Child was terminated by signal", -retcode, file=sys.stderr)
    except OSError as e:
        print("Execution failed:", e, file=sys.stderr)
    print('done')
    

if __name__ == '__main__':
    main()
