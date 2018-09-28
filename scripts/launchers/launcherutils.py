#!/lvol/sfleisch/anaconda3/bin/python3
import sys
import datetime, time
import shlex
import re
import os
import traceback
import subprocess
from functools import partialmethod

class launcherutils(object):
    #Final error check of the logfile after the run.
    #Ignored if the framework doesn't match
    error_pats={
        'caffe2': "^__results.time__",
        'tensorflow': "(?:ResourceExhaustedError|core dumped|std::bad_alloc)",
        'caffe': "Check failure stack trace: \*\*\*",
        'mxnet': "^__results.time__"
    }

    def __init__(self,cmd_args):
        self.vdict=dict([(re.sub('-','_',re.sub('^--','',k)),v) for (k,v) in zip(*[iter(cmd_args[1:])]*2)])
        # Batch information
        try:
           self.logfile=open(self.vdict['exp_log_file'],'a')
        except IOError as e:
           print('Error: ',e)
           sys.exit(-1)

        self.loginfo(' '.join(cmd_args)) # Log command line arguments for debugging purposes
        if not self.check_key('runtime_launcher'): self.vdict['runtime_launcher']=''

        #This will raise an error that we aren't catching if it fails.
        self.assert_not_docker_and_singularity(self.logfatal)
        # If simulate we just want to eventually create and print the script without testing.
        if self.vdict['exp_status'] == 'simulate': return
        self.docker=False
        self.singularity=False
        if self.test_for_true('exp_singularity',self.logfatal):
            self.assert_singularity_image_exists()
            self.singularity=True
        elif self.test_for_true('exp_docker',self.logfatal):
            self.assert_docker_image_exists()
            self.docker=True
        self.__batch_file__ = os.path.join(\
                            os.path.dirname(self.vdict['exp_log_file']),
                                            "{}_{}_{}.batch".format(
                                                self.vdict['exp_framework'],
                                                self.vdict['exp_device_type'],
                                                self.vdict['exp_model']))
        if not self.is_batch_good():
            self.report_and_exit("skipped",
               "The replica batch size ({exp_replica_batch}) is too large for given SW/HW configuration.".format(
                    exp_replica_batch=self.vdict['exp_replica_batch']))

        # Set the error check at the end of the run:
        try:
            self.error_pat=self.__class__.error_pats[self.vdict['exp_framework']]
        except KeyError:
            self.logwarn('Do not have a final error checking regex pattern for the framework: {}'.\
                  format(self.vdict['exp_framework']))
            self.error_pat=None

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

        if self.check_key('exp_mpirun_hosts'):
            self.mpirun_cmd += self.vdict['exp_mpirun_args'] + " -H {} ".format(self.vdict['exp_mpirun_hosts'])
        else:
            self.mpirun_cmd += " "+self.vdict['exp_mpirun_args']
        if not self.check_key('exp_mpirun_num_tasks'): num_tasks=1
        else: num_tasks=self.vdict['exp_mpirun_num_tasks']
        self.mpirun_cmd += " -np {} ".format(self.vdict['exp_mpirun_num_tasks'])

    def grep(self,s):
        fn=self.logfile.name
        self.logfile.close()    
        with open(fn,'r') as r:
            for l in r:
                l=l.strip()
                if re.search(pat,l):
                    found=True
                    break
            else:
                found=False
        self.logfile=open(fn,'a')
        return found

    def is_batch_good(self):
        if not os.path.isfile(self.__batch_file__): 
            #self.logwarn('is_batch_good __batch_file__ {} no file found.'.format(self.__batch_file__))
            return True # File does not exist, no error information
        else:
            with open(self.__batch_file__,'r') as bf:
                self.current_batch=int(bf.readline().strip())
        if int(self.vdict['exp_replica_batch']) < self.current_batch: ret=True # File exists, error batch is larger
        else: ret=False        # File exists, error batch is smaller
        #self.logwarn('is_batch_good current_batch {} exp_replica_batch {} return status {}'.format(self.current_batch,self.vdict['exp_replica_batch'],ret))
        return ret

    def update_error_file(self):
        if not os.path.isfile(self.__batch_file__):
            with open(self.__batch_file__,'w') as bf: print(self.vdict['exp_replica_batch'],file=bf)
        else:
            with open(self.__batch_file__,'r') as bf: current_batch=int(bf.readline().strip())
            if int(self.vdict['exp_replica_batch']) < current_batch:
                with open(self.__batch_file__,'w') as bf: print(self.vdict['exp_replica_batch'],file=bf)

    def report_and_exit(self,status,status_message):
        print("report_and_exit",status, status_message)
        print('__exp.status__= "{}"'.format(status),file=self.logfile)
        print('__exp.status_msg__= "{}"'.format(status_message),file=self.logfile)
        self.logfatal('{} (status code = {})'.format(status_message,status))
        sys.exit(-1)

    #Check the log for fatal errors.  #See above.
    def _error(self):
        if self.error_pat is None: return
        fn=self.logfile.name
        self.logfile.close()
        with open(fn,'r') as r:
            for l in r:
                l=l.strip()
                if re.search(self.error_pat,l):
                    found=True
                    break
            else:
                found=False
        self.logfile=open(fn,'a')
        return found

    # All of the scripts had the same code except for the error function which only differed by the pattern to search for. SO
    # I encapsulated it here.
    def check_for_failed_run(self):
        if self._error():
            self.logwarn('error in "{exp_log_file}" with effective batch {exp_effective_batch} (replica batch {exp_replica_batch})'.\
               format(exp_log_file=self.vdict['exp_log_file'],
                      exp_effective_batch=self.vdict['exp_effective_batch'],
                      exp_replica_batch=self.vdict['exp_replica_batch']))
    
            self.update_error_file()
            print('__exp.status__="failure"',file=self.logfile)

    def run(self,env_command,run_command,benchmark_command,close_log=True):
        script=\
                r'export {framework_env}; '.format(framework_env=env_command)+ \
                r'echo -e "__results.start_time__= \x22$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22"; '+ \
                r'{run_command} {benchmark_command}'.format(run_command=run_command, benchmark_command=benchmark_command)+ \
                r'echo -e "__results.end_time__= \x22$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22"'

        if self.vdict['exp_status']=='simulate':
            print(script)
            sys.exit(0)
        proc=subprocess.Popen(script,executable="/bin/bash",shell=True,stdout=self.logfile,stderr=self.logfile)
        proc.communicate()

        ## Do some post-processing
        self.check_for_failed_run()
        if close_log: self.logfile.close()

