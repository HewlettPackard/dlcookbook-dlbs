#!/lvol/sfleisch/anaconda3/bin/python3
import sys
import datetime, time
import shlex
import re
import os
import traceback
import subprocess
from functools import partialmethod

class commonLauncherClass(object):
    def __init__(self,cmd_args):
        self.vdict=dict([(re.sub('-','_',re.sub('^--','',k)),v) for (k,v) in zip(*[iter(cmd_args[1:])]*2)])
        # Check batch is small enough for this experiment
        self.__batch_file__=os.path.join(os.path.dirname(self.vdict['exp_log_file']),"{}_{}_{}.batch".format(
                                self.vdict['exp_framework'],self.vdict['exp_device_type'],self.vdict['exp_model']))
        try:
           self.logfile=open(self.vdict['exp_log_file'],'a')
        except IOError as e:
           print('Error: ',e)
           sys.exit(-1)

        self.loginfo(' '.join(cmd_args)) # Log command line arguments for debugging purposes
        if not self.check_key('runtime_launcher'): self.vdict['runtime_launcher']=''

        self.assert_not_docker_and_singularity(self.logfatal)
        self.docker=False
        self.singularity=False
        if self.test_for_true('exp_singularity',self.logfatal):
            self.assert_singularity_image_exists()
            self.singularity=True
        elif self.test_for_true('exp_docker',self.logfatal):
            self.assert_docker_image_exists()
            self.docker=True

    def logfileout(self, logtype, s):
        timestamp=datetime.datetime.now().strftime('%m-%d-%Y %H:%M:%S')
        print('{} [{}] {}'.format(timestamp,logtype,s),file=self.logfile)
    
    loginfo=partialmethod(logfileout,'INFO')
    logwarn=partialmethod(logfileout,'WARNING')
    logerr=partialmethod(logfileout,'ERROR')
    logfatal=partialmethod(logfileout,'FATAL')

    def check_key(self,x):
        if x not in self.vdict: return False
        elif self.vdict[x] is None or str(self.vdict[x]).strip()=='': return False
        else: return True
    
    def test_for_true(self,x,loggingfunc):
        if not self.check_key(x): return False
        val= str(self.vdict[x]).lower()
        if val not in ['true','false']:
            loggingfunc("Error in 'test_for_true()', the value of parameter, {}, was neither 'true' or 'false' after case conversion.".format(x))
            raise ValueError()
        return val=='true'
    
    def assert_docker_image_exists(self):
        if not self.check_key('exp_docker_image'):
            self.logfatal('"exp.docker" was "true" but "exp.docker_image" was either missing or blank in the parameter input.')
            raise ValueError()
        p=subprocess.Popen("docker images -q {}".format(self.vdict['exp_docker_image']),shell=True,stdout=subprocess.PIPE)
        s=p.stdout.read().decode("utf-8").strip()
        p.communicate()
        if len(s)>0:
           return True
        else:
           self.logfatal('The Docker image "{}" does not exist locally, pull it from a hub or build it manually'.format(image))
           raise ValueError()
    
    def assert_singularity_image_exists(self):
        if not self.check_key('exp_singularity_image'):
            self.logfatal('"exp.singularity" was "true" but "exp.singularity_image" was either missing or blank in the parameter input.')
            raise ValueError()
        if os.path.isfile(self.vdict['exp_singularity_image']):
            return True
        else:
            self.logfatal('The Singularity image "{}" does not exist locally, pull it from a hub or build it manually'.format(self.vdict['singularity']))
            raise ValueError()
    
    def assert_not_docker_and_singularity(self,loggingfunc):
       if self.test_for_true('exp_docker',self.logfatal) and self.test_for_true('exp_singularity',self.logfatal):
           loggingfunc("Both exp.docker and exp.singularity were set to true, however, only one container type can be selected.")
           raise ValueError()
       else:
           return True

    def setup_mpirun(self):
        try:
             self.mpirun_cmd=self.vdict['exp_mpirun']
        except Exception:
             self.logfatal('exp.mpirun is missing or empty and MPI was specified.')
             raise ValueError()

        if self.check_key('exp_mpirun_hosts'): self.mpi_run_cmd += self.vdict['exp_mpirun_args']) + " -H {} ".format(self.vdict['exp_mpirun_hosts'])
        if not self.check_key('exp_mpirun_num_tasks'): num_tasks=1
        else: num_tasks=self.vdict['exp_mpirun_num_tasks']
        self.mpirun_cmd += " -np {} ".format(self.vdict['exp_mpirun_num_tasks'])

    def run(self,script):
        proc=subprocess.Popen(script,executable="/bin/bash",shell=True,stdout=self.logfile,stderr=self.logfile)
        proc.communicate()
