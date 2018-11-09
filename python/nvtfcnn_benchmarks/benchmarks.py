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

def main():
    """Main worker function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, required=True, default='',
        help="A model to benchmark ('alexnet', 'googlenet' ...)"
    )

    parser = argparse.ArgumentParser(add_help=False)
    #Update here if list of available models change.
    models=['alexnet','inception_resnet_v2','inception_v4','resnet','vgg',
            'googlenet','inception_v3','overfeat','trivial','xception']
    parser.add_argument( '--model', type=str, required=True,
                         default='', choices=models,
                         help="A model to benchmark ({})".format(', '.join(models)))
    args, passthru = parser.parse_known_args()
    m=re.match("(resnet|vgg)_(\d+)",args.model)
    if m:
        args.model=m.group(1)
        passthru+=" --layers {}".format(m.group(2))

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
