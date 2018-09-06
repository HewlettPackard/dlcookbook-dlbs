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

    prog=pathlib.Path(__file__).parent.joinpath('cnn').joinpath(args.model+".py")
    cmd=['python',prog]+passthru
    try:
        retcode = call(cmd)
        if retcode < 0:
            print("Child was terminated by signal", -retcode, file=sys.stderr)
    except OSError as e:
        print("Execution failed:", e, file=sys.stderr)
    print('done')
    
#    if times.size > 0:  
#        times = 1000.0 * times                                              # from seconds to milliseconds
#        mean_time = np.mean(times)                                          # average time in milliseconds
#        mean_throughput = get_effective_batch_size(opts) / (mean_time/1000) # images / sec
#        print("__results.time__=%s" % (json.dumps(mean_time)))
#        print("__results.time_std__=%s" % (json.dumps(np.std(times))))
#        print("__results.throughput__=%s" % (json.dumps(int(mean_throughput))))
#        print("__exp.model_title__=%s" % (json.dumps(model_title)))
#        print("__results.time_data__=%s" % (json.dumps(times.tolist())))
#    else:
#        print("__results.status__=%s" % (json.dumps("failure")))


if __name__ == '__main__':
    main()
