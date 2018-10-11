#!/usr/bin/env python
import sys
import traceback
import re
import enum
import subprocess
import os
import fnmatch
from shutil import copyfile
import argparse

#These will go into a dlbs/lib directory.
def logwarn(logfile, s):
    timestamp=datetime.datetime.now().strftime('%m-%d-%Y %H:%M:%S')
    s=s.strip()
    print('{} [WARNING] {}'.format(timestamp,logtype,s),file=self.logfile)

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

class GrepRet(enum.Enum):
    first=enum.auto()
    last=enum.auto()
    all=enum.auto()

def grep(file=None, pat=None, group=0, split=None, splitOn=' ', occurence=GrepRet.first):
    try:
        fn=file.name
        file.close()
    except Exception:
        fn=file
    with open(fn,'r') as file:
        found=[]
        for l in file:
            m=re.search(pat,l)
            if m:
                if group==0:
                   f=l.strip()
                else:
                   f=m.group(group).strip()
                if split is not None: f=f.split(splitOn)[split-1]
                found.append(f)

                if occurence==GrepRet.first: break
    file.close()
    file=open(fn,'a')
    if len(found)==0: return None,file
    elif occurence==GrepRet.first: return found[0],file
    elif occurence==GrepRet.last: return found[-1],file
    else: return found,file

def caffe_postprocess_log(exp_log_file,exp_device_batch,exp_phase, exp_num_batches, exp_effective_batch):
    #This needs to go into a benchmark.py for Caffe
    with open(exp_log_file,"a") as logfile:
        if self._error():
            logwarn(logfile,'error in "{}" with effective batch {} (per device batch {})'.\
                 format(exp_log_file,effective_batch,exp_device_batch))
            self.update_error_file()
    
            error_hint,logfile=grep(logfile,"Check failed: .*",self.GrepRet.last)
            print('__exp.status__= "failure"', file=logfile)
            print('__exp.status_msg__= "Error has been found in Caffe log file ({})."'.format(error_hint),file=logfile)
            return False
        else:
            # Here we know if we are in time or train mode.
            if exp_phase== "inference":
                elapsed_time,logfile = float(grep(r,'Average Forward pass:',group=0,split=8,occurence=self.GrepRet.last))
            else:
                try:
                    start_time,logfile=grep(logfile,'^I(\d\d\d\d .*?) .*Solver ',group=1,
                                      occurence=self.GrepRet.last)
                    start_time=self.gettimestamp(start_time)
    
                    end_time,logfile=grep(logfile,'^I(\d\d\d\d .*?) .*Optimization Done',group=1,
                                      occurence=self.GrepRet.last)
                    end_time=self.gettimestamp(end_time)
                    elapsed_time=1000.0*(end_time-start_time)/exp_num_batches
                except Exception as e:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
                    logwarn(logfile,'Exception {} Was not able to calculate the execution time from the log file.'.format(type(e)))
                    return
            throughput=1000*exp_effective_batch/elapsed_time
            print('__results.time__= {}'.format(elapsed_time),file=logfile)
            print('__results.throughput__= {}'.format(throughput),file=logfile)
        return True

def caffe_bench(
                dlbs_root=None,
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
        # Make sure model exists
        host_model_dir='{}/python/caffe_benchmarks/models/{}'.format(dlbs_root, exp_model)
        model_file=findfiles("{host_model_dir}/".format(host_model_dir=host_model_dir),
                                "*.{exp_phase}.prototxt".format(exp_phase=exp_phase))[-1]
        caffe_model_file_path =  os.path.join(host_model_dir, caffe_model_file)
        caffe_solver_file_path = os.path.join(host_model_dir,caffe_solver_file)
    
        with open(exp_log_file,'a') as logfile:
             print('__exp.framework_title__="TensorFlow"',file=logfile)
    
        if not os.path.isfile(model_file):
            report_and_exit("failure","A model file ({model_file}) does not exist.".format(model_file=model_file))
            raise(ValueError)
        remove_files([caffe_model_file_path,caffe_solver_file_path])
        copy_files([(model_file, caffe_model_file_path)])
        if exp_phase == "training":
            if exp_data_dir == "":
                sed(caffe_model_file_path, "^#synthetic","")
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
            caffe_solver=re.split('\\\\n',caffe_solver,re.MULTILINE)
            with open(caffe_solver_file_path,'w') as w:
                for l in caffe_solver: print(l,file=w)
        net_name,_=grep(file=caffe_model_file_path,pat='^name: +"(.*?)"',group=1,occurence=GrepRet.first)
    
        print('__exp.model_title__= "{net_name}"'.format(net_name=net_name))
    
        benchmark_command="caffe {caffe_action} {caffe_args}".format(caffe_action=caffe_action,caffe_args=caffe_args)
        # spawn command here
        print('benchmark command: ',benchmark_command)
        #post process the log
        #caffe_postprocess_log(exp_log_file,exp_device_batch,exp_phase, exp_num_batches, exp_effective_batch)
    except Exception as e:
        print('Caught exception. Exiting.')
        traceback.print_exc()
        sys.exit(-1)
    finally:
        pass
        #remove_files([caffe_model_file_path, caffe_solver_file_path])

def main():
    for arg in sys.argv:
        print(arg)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dlbs_root',type=str, required=True, default='',help="dlbs_root")
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
    print('In caffe_benchmarks/benchmarks.py')
    caffe_bench(**dict(args._get_kwargs()))


if __name__=="__main__":
    main()
