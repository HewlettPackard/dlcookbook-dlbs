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
""" Validator verifies that every experiment in a plan can be ran on a machine.

It uses several huristics for that. This is what is checked:

1. `Log files collision check`: no two or more experiments write results in a same file
2. `Docker/NVIDIA docker availability check`: if docker or nvidia docker is required to
   run certain experiments, it checks they are available.
3. `Docker images availability check`: for every docker image, validator checks that
   image exists.
4. `Host framework check`: for a number of frameworks, if they are to run in a host OS,
   validator checks it can do that with provided environmental variables.

Usage:
::

  validator = Validator(plan)  # Create validator.
  validator.validate()         # Compute variables and perform checks.
  validator.report()           # Report results.

"""
import json
import os
import subprocess
import string
import copy
from collections import defaultdict
from dlbs.processor import Processor

class Validator(object):
    """ Validates plan for various errors like log files collision, availability of docker images etc."""

    def __init__(self, plan):
        self.plan = copy.deepcopy(plan)    # Plan - we will compute variables here - so it's a machine dependent validation.
        self.plan_ok = True                # Global summary: is plan OK.
        self.log_files_collisions = set()  # Files that are produced by 2 or more experiments (collisions)
        self.num_disabled = 0              # Number of disabled experiments
        self.frameworks = {}               # Frameworks stats excluding disabled experiments
        self.need_docker = False           # Does the plan need docker (CPU experiments)?
        self.need_nvidia_docker = False    # Does the plan need nvidia docker (GPU experiments)?
        self.cpu_docker_imgs = {}          # The mapping "framework -> set of docker images" for CPU experiments
        self.gpu_docker_imgs = {}          # The mapping "framework -> set of docker images" for GPU experiments
        self.errors = []                   # All the errors found in this plan.
        self.messages = []                 # Any informational message that we want to print in a summary report.
                                           # Like docker images IDs or framework versions.

        self.framework_host_checks = {}    # Temporary storage for the mapping "framework -> env". Env is the env
                                           # variables that we have already checked.

    def validate(self, compute_variables=True):
        """Performs all checks for provided plan.
        
        :param bool compute_variables: If true, variables need to be comptued.
        """
        if compute_variables:
            Processor().compute_variables(self.plan)

        log_files = set()
        for experiment in self.plan:
            # Check log files for collision
            log_file = experiment['exp.log_file']
            if log_file in log_files:
                self.log_files_collisions.add(log_file)
            log_files.add(log_file)
            # Check if experiment is disabled
            if 'exp.disabled' in experiment and experiment['exp.disabled'] == 'true':
                self.num_disabled += 1
            # Update framework statistics
            self.update_framework_stats(experiment)

        # Check docker and nvidia docker installed
        if self.need_docker:
            self.check_can_run_docker(nvidia_docker=False)
        if self.need_nvidia_docker:
            self.check_can_run_docker(nvidia_docker=True)

        # Check images exist
        if self.need_docker or self.need_nvidia_docker:
            for framework in self.frameworks:
                for docker_img in self.frameworks[framework]['docker_images']:
                    self.check_docker_image_exists(docker_img)

        # Set plan OK flag
        if len(self.log_files_collisions) > 0 or len(self.errors) > 0:
            self.plan_ok = False



    def report(self):
        """Prints validation summary."""
        print("====================== VALIDATION REPORT =======================")
        if self.messages:
            print("=========================== MESSAGES ===========================")
            print(json.dumps(list(self.messages), sort_keys=False, indent=4))
        print("======================== FRAMEWORK STATS =======================")
        print(json.dumps(self.frameworks, sort_keys=False, indent=4))
        if not self.plan_ok:
            print("============================ ERRORS ============================")
            if self.log_files_collisions:
                print("Log files collisions (each of these files is the output of at least 2 experiments):")
                print(json.dumps(list(self.log_files_collisions), sort_keys=False, indent=4))
            if self.errors:
                print("Other errors:")
                print(json.dumps(self.errors, sort_keys=False, indent=4))
        print("========================= PLAN SUMMARY =========================")
        print("Is plan OK .......................... %s" % (str(self.plan_ok)))
        print("Number of experiments ............... %d" % (len(self.plan)))
        print("Number of disabled experiments ...... %d" % (self.num_disabled))
        print("Log files collisions ................ %s" % ('YES' if self.log_files_collisions else 'NO'))
        print("================================================================")

    def update_framework_stats(self, exp):
        """Updates statistics for a framework in this experiment `exp`.

        Will not update stats if this experiment is disabled.

        :param dict exp: An experiment item from :py:meth:`~dlbs.Validator.plan` list.
        """
        framework_id = exp['exp.framework_id']
        if framework_id not in self.frameworks:
            self.frameworks[framework_id] = {
                'num_exps': 0,
                'num_docker_exps': 0,
                'num_host_exps': 0,
                'num_disabled': 0,
                'num_gpu_exps': 0,
                'num_cpu_exps': 0,
                'docker_images': []
            }
        stats = self.frameworks[framework_id]
        # Update number of disabled exps
        if 'exp.disabled' in exp and exp['exp.disabled'] == 'true':
            stats['num_disabled'] += 1
            self.num_disabled += 1
            return
        stats['num_exps'] += 1
        # Update docker/host stats
        docker_img_key = ""
        if exp['exp.env'] == 'docker':
            stats['num_docker_exps'] += 1
            docker_img_key = '%s.docker.image' % (exp['exp.framework'])
            if exp[docker_img_key] not in stats['docker_images']:
                stats['docker_images'].append(exp[docker_img_key])
        elif exp['exp.env'] == 'host':
            stats['num_host_exps'] += 1
            self.check_host_framework(exp['exp.framework'], exp['%s.env' % (exp['exp.framework'])])
        else:
            self.errors.append("Unknown experiment environment: '%s'. Expecting 'docker' or 'host'" % (exp['exp.env']))
        # Update CPU/GPU stats
        if exp['exp.device'] == 'gpu':
            stats['num_gpu_exps'] += 1
            if exp['exp.env'] == 'docker':
                self.need_nvidia_docker = True
                self.add_docker_image(exp['exp.framework'], 'gpu', exp[docker_img_key])
        elif exp['exp.device'] == 'cpu':
            stats['num_cpu_exps'] += 1
            if exp['exp.env'] == 'docker':
                self.need_docker = True
                self.add_docker_image(exp['exp.framework'], 'cpu', exp[docker_img_key])
        else:
            stats['errors'].append("Unknown device: '%s'. Expecting 'gpu' or 'cpu'" % (exp['exp.device']))

    def add_docker_image(self, framework, device, docker_img):
        """Adds CPU or GPU docker image to list of images.

        :param str framework:  Framework name (caffe, tensorflow, caffe2 ...). This\
                               is not a framework ID, so, you should not pass here something like\
                               bvlc_caffe - use caffe for all caffe forks.
        :param str device:     Main computational device: 'cpu' or 'gpu'. Based on this\
                               value we select docker/nvidia-docker.
        :param str docker_img: Name of a docker image.
        """
        imgs = self.gpu_docker_imgs if device == 'gpu' else self.cpu_docker_imgs
        if framework not in imgs:
            imgs[framework] = set()
        imgs[framework].add(docker_img)

    def check_host_framework(self, framework, env):
        """Checks it can run framework in host environment.

        :param str framework: Framework name (caffe, tensorflow, caffe2 ...). This\
                              is not a framework ID, so, you should not pass here something like\
                              bvlc_caffe - use caffe for all caffe forks.
        :param str env: Environmental variables from experiment. We will parse them\
                        into a dictionary to pass to a`subprocess.Popen` call.
        """
        # Check if we have already performed this test
        if framework not in self.framework_host_checks:
            self.framework_host_checks[framework] = set()
        elif env in self.framework_host_checks[framework]:
            return

        self.framework_host_checks[framework].add(env)

        # Convert `env` to valid dictionary of variables. Here, the 'env' is a
        # computed parameter. It still may have dependencies on standard system
        # variables like PYTHONPATH or LD_LIBRARY_PATH. We need to take care if
        # these variables are not in os.environ when we substitute them. That's
        # defaultdict is used here.
        variables = {}
        for item in string.Template(env.replace('\$', '$')).substitute(defaultdict(lambda: '', os.environ)).split():
            name_value = item.split('=')
            variables[name_value[0]] = name_value[1]
        # Run simple tests
        cmd = None
        if framework == 'tensorflow':
            cmd = ['python', '-c', 'import tensorflow as tf; print tf.__version__;']
        elif framework == 'mxnet':
            cmd = ['python', '-c', 'import mxnet as mx; print mx.__version__;']
        elif framework == 'caffe2':
            # It seems that in the future releases it'll be possible to use something like:
            # from caffe2.python.build import build_options
            #cmd = ['python', '-c', 'import caffe2;']
            cmd = ['python', '-c', 'from caffe2.python.build import build_options; print(build_options);']
            #cmd = ['python', '-c', 'from caffe2.python.build import CAFFE2_NO_OPERATOR_SCHEMA; print(CAFFE2_NO_OPERATOR_SCHEMA);']
        elif framework == 'caffe':
            cmd = ['caffe', '--version']
        elif framework == 'tensorrt':
            cmd = ['tensorrt', '--version']

        if cmd is not None:
            retcode, output = Validator.run_process(cmd, env=variables)
            self.add_check_result('CheckHostFramework', cmd, retcode, output, env=variables)

    def check_can_run_docker(self, nvidia_docker=False):
        """Checks it can run docker/nvidia-docker.

        :param bool nvidia_docker: If True, `nvidia-docker` should be used. Else `docker`.
        """
        try:
            if nvidia_docker:
                cmd = ["nvidia-docker", "--version"]
            else:
                cmd = ["docker", "--version"]
            retcode, output = Validator.run_process(cmd)
            self.add_check_result('CanRunDocker', cmd, retcode, output)
        except OSError as error:
            self.add_check_result('CanRunDocker', cmd, retcode, output, error=error)
            #self.errors.append("CanRunDocker (nvidia_docker=%s) check failed with message: '%s'" % (nvidia_docker, str(error)))

    def check_docker_image_exists(self, docker_img):
        """Checks if this docker image exists.

        :param str docker_img: Name of a docker image.
        """
        try:
            cmd = ["docker", "inspect", "--type=image", docker_img]
            retcode, output = Validator.run_process(cmd)
            # The 'docker inspect ...' will return 0 if image exists and 1 otherwise.
            # We should also ignore output - it's a bunch of information on this image
            self.add_check_result(
                'DockerImageExists',
                cmd,
                retcode,
                output if retcode != 0 else ['docker image exists']
            )
        except OSError as error:
            self.errors.append("DockerImageExists (image=%s) check failed with message: '%s'" % (docker_img, str(error)))

    @staticmethod
    def run_process(cmd, env=None):
        """Runs process with subprocess.Popen (some kind of test).

        :param list cmd: A command with its arguments to run.
        :param dict env: Environmental variables to initialize environment.
        :return: tuple (return_code (int), command_output (list of strings))
        """
        process = subprocess.Popen(cmd, universal_newlines=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env=env)
        output = []
        while True:
            line = process.stdout.readline()
            if line == '' and process.poll() is not None:
                break
            if line:
                output.append(line)
        #print("%d -> %s" % (process.returncode, str(output)))
        return (process.returncode, output)

    def add_check_result(self, check_name, cmd, retcode, output, **kwargs):
        """Adds check result to list of messages/errors depending on return code.

        :param str check_name: Name of this check, something like `DockerImageExists`.
        :param list cmd: a command with its arguments to run.
        :param int retcode: A return code. The '0' value most likely indicates OK.
        :param list output: A lsit of strings - output of the command run.
        :param kwargs: Any other named parameters to add to a report.
        """
        message = {
            'check_name': check_name,
            'cmd': ' '.join(cmd),
            'output': ' '.join(output),
            'retcode': retcode
        }
        for key in kwargs:
            message[key] = kwargs[key]

        if retcode == 0:
            self.messages.append(message)
        else:
            self.errors.append(message)
