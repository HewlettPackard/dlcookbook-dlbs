"""
Proof of concept - work in progress.
Implemented: [this seems to be a preferred choice]
    - Packaged DLBS with external configuration.

To be done:
    - Packaged DLBS with internal configuration.
"""
import os
import stat
import shutil
import argparse


class MLBoxBuilder(object):
    """
    TODO: Move script contents to template files and process them with DLBS's processor, like benchmark configurations.
    TODO: This will enable users to provide their own templates any parameters.
    """
    def __init__(self, args):
        # FIXME: deep copy or something ...
        self.args = args
        self.work_dir = os.path.abspath(self.args.work_dir)

    def create_directory_structure(self):
        if os.path.exists(self.work_dir):
            raise RuntimeError("Working directory must not exist({})".format(self.work_dir))

        os.makedirs(self.work_dir)
        for folder in ('config', 'logs', 'mlbox'):
            os.makedirs(os.path.join(self.work_dir, folder))

    @staticmethod
    def make_executabe(file_name):
        st = os.stat(file_name)
        os.chmod(file_name, st.st_mode | stat.S_IEXEC)

    def create_docker_file(self):
        docker_file = os.path.join(self.work_dir, 'mlbox', 'Dockerfile')
        with open(docker_file, mode='w') as fobj:
            fobj.write("FROM {}\n".format(self.args.base_image))
            fobj.write("\n# DLBS hash tag\n")
            fobj.write("ARG DLBS_HASHTAG='{}'\n".format(self.args.hashtag))
            fobj.write("\n# Config files need to reference these variables if they"
                       " want to be compatible with the MLBox\n")
            fobj.write("ENV CONFIG_ROOT='/workspace/config'\n")
            fobj.write("ENV LOGS_ROOT='/workspace/logs'\n")
            fobj.write("ENV DATA_ROOT='/workspace/data'\n")
            fobj.write("\n# Clone DLBS\n")
            fobj.write("RUN git clone https://github.com/HewlettPackard/dlcookbook-dlbs.git /dlbs &&\n")
            fobj.write("    cd /dlbs && git reset --hard ${DLBS_HASHTAG}\n")
            fobj.write("\n# Initialize environment and run packaged configuration\n")
            fobj.write("ENTRYPOINT /bin/bash -c \"source /dlbs/scripts/environment.sh && \\\n")
            fobj.write("                         python /dlbs/python/dlbs/experimenter.py run "
                       "--config /${CONFIG_ROOT}/config.json\"\n")

    def create_build_script(self):
        build_script = os.path.join(self.work_dir, 'build.sh')
        with open(build_script, mode='w') as fobj:
            fobj.write("#!/bin/bash\n")
            fobj.write("\n# User-provided docker image name\n")
            fobj.write("img_name=\"{}\"\n".format(self.args.docker_image))
            fobj.write("\n# Runtime build arguments\n")
            fobj.write("args=\"\"\n")
            fobj.write("[[ -n \"${http_proxy}\" ]] && args=\"${args} --build-arg http_proxy=${http_proxy}\"\n")
            fobj.write("[[ -n \"${https_proxy}\" ]] && args=\"${args} --build-arg https_proxy=${https_proxy}\"\n")
            fobj.write("docker build -t ${img_name} ${args} ./mlbox\n")
        MLBoxBuilder.make_executabe(build_script)

    def create_run_script(self):
        build_script = os.path.join(self.work_dir, 'run.sh')
        with open(build_script, mode='w') as fobj:
            fobj.write("#!/bin/bash\n")
            fobj.write("\nexport ROOT_DIR=$( cd $( dirname \"${BASH_SOURCE[0]}\" ) && pwd )\n")

            fobj.write("\ndocker_img=\"{}\"    # User-provided docker image name.\n".format(self.args.docker_image))
            fobj.write("docker_launcher=\"nvidia-docker\"         # This should come from user config.\n")

            fobj.write("\nLOGS_ROOT=\"/workspace/logs\"\n")
            fobj.write("CUDA_CACHE=\"/workspace/cuda_cache\"\n")
            fobj.write("CONFIG_ROOT=\"/workspace/config\"\n")

            fobj.write("\ndocker_args=\"-i --security-opt seccomp=unconfined --rm --shm-size=1g --ulimit memlock=-1"
                       " --ulimit stack=67108864 --ipc=host\"\n")
            fobj.write("docker_args=\"${docker_args} --volume ${ROOT_DIR}/logs:${LOGS_ROOT}\"\n")
            fobj.write("docker_args=\"${docker_args} --volume ${ROOT_DIR}/config:${CONFIG_ROOT}\"\n")
            fobj.write("docker_args=\"${docker_args} --volume /dev/shm/dlbs:/workspace/cuda_cache\"\n")

            fobj.write("\n${docker_launcher} run ${docker_args} ${docker_img}\n")
        MLBoxBuilder.make_executabe(build_script)

    def build(self):
        self.create_directory_structure()
        self.create_docker_file()
        self.create_build_script()
        self.create_run_script()
        if os.path.exists(self.args.config):
            shutil.copyfile(self.args.config, os.path.join(self.work_dir, 'config', 'config.json'))


if __name__ == '__main__':
    # TODO: this script, similar to the DLBS main entry point, must accept arbitrary parameters.
    parser = argparse.ArgumentParser(description="DLBS MLBox packager")
    parser.add_argument('--config', required=True, type=str, help="Path to a DLBS JSON configuration file.")
    parser.add_argument('--hashtag', required=True, type=str, help="DLBS Hash tag (GitHub commit).")
    parser.add_argument('--work_dir', required=True, type=str, help="Working directory.")
    parser.add_argument('--docker_image', required=True, type=str, help="Name of an output docker image.")
    parser.add_argument('--base_image', required=True, type=str, help="Name of a base docker image.")
    parser.add_argument('--docker_launcher', required=False, default="docker", type=str, help="Docker executable.")

    builder = MLBoxBuilder(parser.parse_args())
    builder.build()
