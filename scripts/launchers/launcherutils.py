#!/usr/bin/env python
import sys
from datetime import datetime
import shlex
import re
import os
import traceback
import subprocess
import fnmatch
from functools import partialmethod
from shutil import copyfile
import enum

class launcherutils(object):
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
        if 'exp_status' in self.vdict and self.vdict['exp_status'] == 'simulate': return
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

    def logfileout(self, logtype, s):
        timestamp=datetime.now().strftime('%m-%d-%Y %H:%M:%S')
        s=s.strip()
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
           self.logfatal('The Docker image "{}" does not exist locally, pull it from a hub or build it manually'.\
                           format(self.vdict['exp_docker_image']))
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
            self.mpirun_cmd += " {} -H {} ".format(self.vdict['exp_mpirun_hosts'], self.vdict['exp_mpirun_args'])
        else:
            self.mpirun_cmd += " {}".format(self.vdict['exp_mpirun_args'])
        if not self.check_key('exp_mpirun_num_tasks'): num_tasks=1
        else: num_tasks=self.vdict['exp_mpirun_num_tasks']
        self.mpirun_cmd += " -np {} ".format(self.vdict['exp_mpirun_num_tasks'])

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
        fn=self.logfile.name
        self.logfile.close()
        with open(fn,'r') as r:
            for l in r:
                l=l.strip()
                if re.search('^__results.time__=[\d\.]',l):
                    found=True
                    break
            else:
                found=False
        self.logfile=open(fn,'a')
        return not found

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

    def run(self,run_command,benchmark_command,close_log=True):
        env=self.vdict['{}_env'.format(self.vdict['exp_framework'])]
        script=\
                r'{run_command} /bin/bash -c " {env} {runtime_python} {benchmark_command}"'.format(\
						run_command=run_command,env=env,runtime_python=self.vdict['runtime_python'],benchmark_command=benchmark_command)
        if self.vdict['exp_status']=='simulate':
            print(script)
            sys.exit(0)
        proc=subprocess.Popen(script,executable="/bin/bash",shell=True,stdout=self.logfile,stderr=self.logfile,bufsize=1,universal_newlines=False)
        proc.communicate()


        ## Do some post-processing
        self.check_for_failed_run()
        if close_log: self.logfile.close()
    @staticmethod
    def findfiles(root,pat):
        result=[]
        for root,dirs,files in os.walk(root):
            for name in files:
                if fnmatch.fnmatch(name,pat):
                    result.append(os.path.join(root,name))
        return result

    @staticmethod
    def sed(infile,outfile=None,pats=None,count=0,flags=0):
        with open(infile,'r') as r: lines=r.readlines()
        for i,l in enumerate(lines):
            for from_pat,to_pat in pats:
                l=re.sub(from_pat,to_pat,l,count=count,flags=flags)
                lines[i]=l
    
        if outfile is None: outfile=infile
        with open(outfile,'w') as w:
            for l in lines: w.write(l)
    @staticmethod
    def remove_files(files,silent=True):
        for file in files:
            try: os.remove(file)
            except Exception as e:
                if e.errno == 2:
                    pass
                else:
                    co.logfatal('remove_files: file "{}" was not removed.'.format(file))
                    raise(e)
            else:
                if not silent: print("removed {}".format(file))
    @staticmethod
    def copy_files(file_tuples):
        for from_file, to_file in file_tuples:
            try:
                copyfile(from_file, to_file)
            except Exception as e:
                report_and_exit("failure",'Cannot copy "{from_file}" to "{to_file}"'.\
                    format(from_file=from_file,to__file=to__file))
                raise(e)

    @staticmethod
    def gettimestamp(s,fmt="%m%d %H:%M:%S.%f",needyear=True):
        try:
            t=datetime.strptime(s, fmt)
            if needyear: 
                y=datetime.now().year
            else:
                y=t.year
            ts=datetime(y,t.month,t.day,t.hour,t.minute,t.second,t.microsecond)
            ts=ts.timestamp()
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
            raise

        return ts

    class GrepRet(enum.Enum):
        first=enum.auto()
        last=enum.auto()
        all=enum.auto()
    
    @staticmethod
    def grep(file=None, pat=None, group=0, split=None, splitOn=' ', occurence=GrepRet.first, mode='a'):
        fn=file.name
        with open(fn,'r') as file:
            found=[]
            for l in file:
                m=re.search(pat,l)
                if m:
                    if group==0:
                       f=l.strip()
                    else:
                       f=m.group(group).strip()
                    if split is not None:
                       f=f.split(splitOn)[split-1]
                    found.append(f)
                    if occurence==launcherutils.GrepRet.first: break
        file=open(fn,mode)
        if len(found)==0: return None,file
        elif occurence==launcherutils.GrepRet.first: return found[0],file
        elif occurence==launcherutils.GrepRet.last: return found[-1],file
        else: return found,file
