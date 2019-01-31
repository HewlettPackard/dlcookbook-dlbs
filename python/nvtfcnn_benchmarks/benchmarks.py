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
import os
from os import stat
from pwd import getpwuid
from glob import glob
from shutil import rmtree
import argparse
import pathlib
import sys
import re
import subprocess
import traceback

class RE():
    # Use re in if tests with side effect.
    def search(self,*args,**kwargs):
        self.s=re.search(*args,*kwargs)
        return self.s
    def match(self,*args,**kwargs):
        self.m=re.match(*args,*kwargs)
        return self.m

def cleanup_tmp():
    user=getpwuid(os.getuid()).pw_name
    for t in glob('/tmp/tmp*'):
        try:
            st=os.stat(t)
            userinfo = getpwuid(st.st_uid).pw_name
            if userinfo==user: rmtree(t)
        except IOException:
            pass

models_re= [r"alexnet$",r"googlenet$",r"inception_resnet2$",r"inception[34]$",r"overfeat$",r"resnet\d+$",r"trivial$",r"vgg\d+$",r"xception$"]

def allowed_models(v):
    v=v.strip()
    for model_re in models_re:
        if re.match(model_re,v): break
    else:
        raise argparse.ArgumentTypeError('The specified model {} does not match any recognized pattern ({})'.format(v,','.join(models_re)))
    return v

def main():
    """Main worker function."""
    try:
        parser = argparse.ArgumentParser(add_help=False)
        #Update here if list of available models change.
        parser.add_argument( '--model', type=allowed_models, required=True,
                             default='', 
                             help="A model to benchmark - must match one of the patterns: {}".format(' | '.join(models_re)))
        #parser.add_argument('--cleanup', action='store_const', default = True, const = True, dest='cleanup')
        #parser.add_argument('--no-cleanup', action='store_const', const = False, dest='cleanup')

        args, passthru = parser.parse_known_args()
        # Transform model names from DLBS conventions to nvtfcnn.
        reobj=RE()
        if reobj.match("(resnet|vgg)(\d+)",args.model):
            args.model=reobj.m.group(1)
            passthru+=["--layers={}".format(reobj.m.group(2))]
        elif reobj.match("inception([34])",args.model):
            args.model="inception_v{}".format(reobj.m.group(1))
        elif reobj.match("inception_resnet2",args.model):
            args.model="inception_resnet_v2"
        prog=pathlib.Path(__file__).parent.joinpath('cnn').joinpath(args.model+".py")
        cmd=['python',str(prog)] + passthru
        print(cmd)
        try:
            subprocess.run(cmd,check=True,universal_newlines=True,stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print("Execution failed:", e)
        #finally:
            #if args.cleanup: cleanup_tmp()
        print('done')
    except Exception as e:
        print('nvtfcnn_benchmarks/benchmark.py: Something failed.')
        print('cmd: {}'.format(' '.join(cmd)))
        traceback.print_exc()
        sys.exit(-1)
    #finally:
    #    if args.cleanup: cleanup_tmp()

if __name__ == '__main__':
    main()
