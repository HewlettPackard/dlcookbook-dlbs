#!/usr/bin/env python
import sys
import traceback
import re
import subprocess
import os
from launcherutils import launcherutils

class caffe_launcherutils(launcherutils):
    def __init__(self,args):
        super(caffe_launcherutils,self).__init__(args)
    def caffe_postprocess_log(self):
        #This needs to go into a benchmark.py for Caffe
        need_to_reopen=False
        try:
           if not self.logfile.closed:
               self.logfile.closed
               need_to_reopen = True
        except Exception:
           pass
        with open(self.vdict['exp_log_file'],"a") as logfile:
            if self._error():
                self.logwarn('error in "{}" with effective batch {} (per device batch {})'.\
                     format(self.vdict['exp_log_file'],self.vdict['exp_effective_batch'],self.vdict['exp_replica_batch']))
                self.update_error_file()
    
                error_hint,logfile=self.grep(logfile,"Check failed: .*",self.GrepRet.last)
                print('__exp.status__= "failure"')
                print('__exp.status_msg__= "Error has been found in Caffe log file ({})."'.format(error_hint))
                return False
            else:
                # Here we know if we are in time or train mode.
                if self.vdict['exp_phase'] == "inference":
                    elapsed_time,logfile = float(self.grep(r,'Average Forward pass:',group=0,split=8,occurence=self.GrepRet.last))
                else:
                    try:
                        start_time,logfile=self.grep(logfile,'^I(\d\d\d\d .*?) .*Solver ',group=1,
                                          occurence=self.GrepRet.last)
                        start_time=self.gettimestamp(start_time)
    
                        end_time,logfile=self.grep(logfile,'^I(\d\d\d\d .*?) .*Optimization Done',group=1,
                                          occurence=self.GrepRet.last)
                        end_time=self.gettimestamp(end_time)
                        elapsed_time=1000.0*(end_time-start_time)/float(self.vdict['exp_num_batches'])
                    except Exception as e:
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        traceback.print_tb(exc_traceback, limit=1, file=logfile)
                        self.logwarn('Exception {} Was not able to calculate the execution time from the log file.'.format(type(e)))
                        return
                throughput=1000*float(self.vdict['exp_effective_batch'])/elapsed_time
                print('__results.time__= {}'.format(elapsed_time),file=logfile)
                print('__results.throughput__= {}'.format(throughput),file=logfile)
            if need_to_reopen: self.logfile=open(self.vdict['exp_log_file'],'a')
        return True
    #Check the log for fatal errors.  #See above.
    def _error(self):
        fn=self.logfile.name
        self.logfile.close()
        with open(fn,'r') as r:
            for l in r:
                l=l.strip()
                if re.search('Optimization Done in \d+',l):
                    found=True
                    break
            else:
                found=False
        self.logfile=open(fn,'a')
        return not found

def main():
    try:
        co=caffe_launcherutils(sys.argv)
        #co.setup_mpirun()
        if co.singularity:
            run_command=\
                r'{runtime_launcher}  {exp_singularity_launcher} exec {caffe_singularity_args} '.format(\
                   runtime_launcher=co.vdict['runtime_launcher'],
                   exp_singularity_launcher=co.vdict['exp_singularity_launcher'],
                   caffe_singularity_args=co.vdict['caffe_singularity_args'])
        elif co.docker:
            run_command=\
                r'{runtime_launcher} {exp_docker_launcher} run {caffe_docker_args}'.format(\
                   runtime_launcher=co.vdict['runtime_launcher'],
                   exp_docker_launcher=co.vdict['exp_docker_launcher'],
                   caffe_docker_args=co.vdict['caffe_docker_args'])
        else:
            run_command=\
                r'{runtime_launcher} {runtime_python}'.format(\
                   runtime_launcher=co.vdict['runtime_launcher'])

        benchmark_command = '{}/benchmarks.py '.format(co.vdict['caffe_bench_path'])
        for param in ['caffe_bench_path','caffe_data_mean_file','caffe_data_mean_file_name','caffe_model_file','caffe_nvidia_backward_math_precision',
             'caffe_nvidia_backward_precision','caffe_nvidia_forward_math_precision','caffe_nvidia_forward_precision','caffe_solver',
             'caffe_solver_file','exp_data_dir','exp_docker','exp_singularity','exp_effective_batch','exp_framework_fork',
             'exp_framework_title','exp_backend','exp_log_file','exp_model','exp_phase','exp_replica_batch','runtime_launcher',
             'caffe_action','caffe_args']:
             if co.vdict[param] == '' or co.vdict[param] is None: arg='\\\"\\\"'
             else:
                 arg=co.vdict[param].strip() #.replace('"','\\\"').replace("'","\\\'")
                 if re.search(' ',arg): arg='\\"{}\\"'.format(arg)
             benchmark_command+=" --{} ".format(param)+arg

        co.run(run_command,benchmark_command)
        co.caffe_postprocess_log()

    except Exception as e:
        print('Caught exception. Exiting.')
        traceback.print_exc()
        sys.exit(-1)
if __name__=="__main__":
    main()
