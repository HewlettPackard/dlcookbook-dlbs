#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
import traceback
import re
from enum import Enum
import subprocess
import os
import fnmatch
from shutil import copyfile
import argparse
import shlex
try:
    from io import StringIO
except Exception:
    from StringIO import StringIO

#These will go into a dlbs/lib directory.
def testforutf8():
    if sys.stdout.encoding.upper() != 'UTF-8':
        enc=sys.stdout.encoding.upper()
        raise ValueError(
           '''
           The encoding for sys.stdout is not UTF-8.
           Export PYTHONIOENCODING at the shell or set it on the command line.
           I.e.,
           export PYTHONIONENCODING=UTF-8
           or,
           PYTHONIONENCODING=UTF-8 {} ....
           '''.format(sys.argv[0]))

# Should change this to use the Template class
def sed(infile,outfile=None,pats=None,count=0,flags=0):
    with open(infile,'r') as r: lines=r.readlines()
    for i,l in enumerate(lines):
        for from_pat,to_pat in pats:
            l=re.sub(from_pat,to_pat,l,count=count,flags=flags)
            lines[i]=l

    if outfile is None: outfile=infile
    with open(outfile,'w') as w:
        for l in lines: w.write(l)

def copy_files(file_tuples):
    for from_file, to_file in file_tuples:
        try:
            copyfile(from_file, to_file)
        except Exception as e:
            print("failure",'Cannot copy "{from_file}" to "{to_file}"'.\
                format(from_file=from_file,to__file=to__file))
            raise(e)

def remove_files(files,silent=True):
    for file in files:
        try: os.remove(file)
        except Exception as e:
            if e.errno == 2:
                pass
            else:
                co.logfatal('remove_files: file "{}" was not removed.'.format(file))
                #print('remove_files: file "{}" was not removed. Errno: {}'.format(file,e.errno))
                raise(e)
        else:
            if not silent: print("removed {}".format(file))

def findfiles(root,pat):
    result=[]
    for root,dirs,files in os.walk(root):
        for name in files:
            if fnmatch.fnmatch(name,pat):
                result.append(os.path.join(root,name))
    return result

def caffe_bench(
                caffe_bench_path=None,
                caffe_data_mean_file=None,
                caffe_data_mean_file_name=None,
                caffe_model_file=None,
                caffe_nvidia_backward_math_precision=None,
                caffe_nvidia_backward_precision=None,
                caffe_nvidia_forward_math_precision=None,
                caffe_nvidia_forward_precision=None,
                caffe_solver=None,
                caffe_solver_file=None,
                exp_data_dir=None,
                exp_docker=None,
                exp_singularity=None,
                exp_effective_batch=None,
                exp_framework_fork=None,
                exp_framework_title=None,
                exp_backend=None,
                exp_log_file=None,
                exp_model=None,
                exp_phase=None,
                exp_replica_batch=None,
                runtime_launcher=None,
                caffe_action=None,
                caffe_args=None,
            ):
    try:
        try: testforutf8()
        except ValueError as e:
            print(e)
            raise

        # Make sure model exists
        host_model_dir='{}/models/{}'.format(caffe_bench_path, exp_model)
        model_file=findfiles("{host_model_dir}/".format(host_model_dir=host_model_dir),
                                "*.{exp_phase}.prototxt".format(exp_phase=exp_phase))[-1]
        caffe_model_file_path =  os.path.join(host_model_dir, caffe_model_file)
        caffe_solver_file_path = os.path.join(host_model_dir,caffe_solver_file)
    
        print('__exp.framework_title__="TensorFlow"')
    
        if not os.path.isfile(model_file):
            report_and_exit("failure","A model file ({model_file}) does not exist.".format(model_file=model_file))
            raise(ValueError)
        remove_files([caffe_model_file_path,caffe_solver_file_path])
        copy_files([(model_file, caffe_model_file_path)])
        if exp_phase == "training":
            if exp_data_dir == "":
                sed(caffe_model_file_path, outfile=None, pats=[("^#synthetic","")])
            else:
                if exp_docker or exp_singularity == "true":
                    real_data_dir="/workspace/data"
                    real_data_mean_file="/workspace/image_mean/{caffe_data_mean_file_name}".\
                       format(caffe_data_mean_file_name=caffe_data_mean_file_name)
                else:
                   real_data_dir=exp_data_dir
                   real_data_mean_file=caffe_data_mean_file
                sed(caffe_model_file_path,outfile=None,pats=[
                    ("^#data",""),
                    ("__CAFFE_MIRROR__",caffe_mirror),
                    ("__CAFFE_DATA_MEAN_FILE__",real_data_mean_file),
                    ("__CAFFE_DATA_DIR__",real_data_dir),
                    ("__CAFFE_DATA_BACKEND__",caffe_data_backend)
                    ])
            if exp_framework_fork == "nvidia":
                sed(caffe_model_file_path,outfile=None,pats=[
                    ("^#precision",""),
                    ("__FORWARD_TYPE___",caffe_nvidia_forward_precision),
                    ("__BACKWARD_TYPE___",caffe_nvidia_backward_precision),
                    ("__FORWARD_MATH___",caffe_nvidia_forward_math_precision),
                    ("__BACKWARD_MATH___",caffe_nvidia_backward_math_precision)
                    ])
        if exp_framework_fork == "nvidia":
            # NVIDIA Caffe - strong scaling for real data and weak scaling for synthetic one
            if exp_data_dir == "":
                # Synthetic data with 'Input' layer - Caffe is in weak scaling model
                sed(caffe_model_file_path,pats=[("__EXP_DEVICE_BATCH__",exp_replica_batch)])
            else:
                # Real data - Caffe is in strong scaling mode - it will divide whatever batch size we have in
                # protobuf by number of solvers.
                sed(caffe_model_file_path,pats=[("__EXP_DEVICE_BATCH__",exp_effective_batch)])
        else:
            # This is for BVLC/Intel Caffe
            sed(caffe_model_file_path,pats=[("__EXP_DEVICE_BATCH__",exp_replica_batch)])
    
        if exp_phase == "training": 
            c= StringIO(caffe_solver.decode('utf-8'))
            with open(caffe_solver_file_path,'w') as w:
                for l in c:
                     print(l.strip(),file=w)
            c.close()
        with open(caffe_model_file_path,'r') as r:
             for line in r:
                 m=re.match('name: +"(.*?)"',line)
                 if m:
                     net_name=m.group(1)
                     break
             else: net_name=''
    
        print('__exp.model_title__= "{net_name}"'.format(net_name=net_name))
    
        benchmark_command="caffe {caffe_action} {caffe_args}".format(caffe_action=caffe_action,caffe_args=caffe_args)
        # spawn command here
        process = subprocess.Popen(shlex.split(benchmark_command), shell=False, bufsize=1,
                                   universal_newlines=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env=os.environ)
        output = []
        while True:
            output= process.stdout.readline()
            pp=process.poll()
            if isinstance(output,bytes): output=output.decode('utf-8')
            if output == '' and (pp == 0 or pp is None): break
            if output !='':
                print(output.strip())
    except Exception as e:
        print('Caught exception. Exiting.')
        traceback.print_exc()
        sys.exit(-1)
    finally:
        pass
        remove_files([caffe_model_file_path, caffe_solver_file_path])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--caffe_bench_path', type=str, required=True, default='',help="caffe_bench_path")
    parser.add_argument('--caffe_data_mean_file',type=str, required=True, default='',help="caffe_data_mean_file")
    parser.add_argument('--caffe_data_mean_file_name',type=str, required=True, default='',help="caffe_data_mean_file_name")
    parser.add_argument('--caffe_model_file',type=str, required=True, default='',help="caffe_model_file")
    parser.add_argument('--caffe_nvidia_backward_math_precision',type=str, required=True, default='',help="caffe_nvidia_backward_math_precision")
    parser.add_argument('--caffe_nvidia_backward_precision',type=str, required=True, default='',help="caffe_nvidia_backward_precision")
    parser.add_argument('--caffe_nvidia_forward_math_precision',type=str, required=True, default='',help="caffe_nvidia_forward_math_precision")
    parser.add_argument('--caffe_nvidia_forward_precision',type=str, required=True, default='',help="caffe_nvidia_forward_precision")
    parser.add_argument('--caffe_solver',type=str, required=True, default='',help="caffe_solver")
    parser.add_argument('--caffe_solver_file',type=str, required=True, default='',help="caffe_solver_file")
    parser.add_argument('--exp_data_dir',type=str, required=True, default='',help="exp_data_dir")
    parser.add_argument('--exp_docker',type=str, required=True, default='',help="exp_docker")
    parser.add_argument('--exp_singularity',type=str, required=True, default='',help="exp_docker")
    parser.add_argument('--exp_effective_batch',type=str, required=True, default='',help="exp_effective_batch")
    parser.add_argument('--exp_framework_fork',type=str, required=True, default='',help="exp_framework_fork")
    parser.add_argument('--exp_framework_title',type=str, required=True, default='',help="exp_framework_fork")
    parser.add_argument('--exp_backend',type=str, required=True, default='',help="exp_framework_fork")
    parser.add_argument('--exp_log_file',type=str, required=True, default='',help="exp_log_file")
    parser.add_argument('--exp_model',type=str, required=True, default='',help="exp_model")
    parser.add_argument('--exp_phase',type=str, required=True, default='',help="exp_phase")
    parser.add_argument('--exp_replica_batch',type=str, required=True, default='',help="exp_replica_batch")
    parser.add_argument('--runtime_launcher',type=str, required=True, default='',help="runtime_launcher")
    parser.add_argument('--caffe_action',type=str, required=True, default='',help="caffe_action")
    parser.add_argument('--caffe_args',type=str, required=True, default='',help="caffe_args")
    args = parser.parse_args()
    caffe_bench(**dict(args._get_kwargs()))


if __name__=="__main__":
    main()
