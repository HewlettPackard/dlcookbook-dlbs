#!/bin/bash

export ROOT_DIR=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )

docker_img="dlbs/ngc-tf:18.07-mlbox"    # User-provided docker image name.
docker_launcher="nvidia-docker"         # This should come from user config.

LOGS_ROOT="/workspace/logs"
CUDA_CACHE="/workspace/cuda_cache"
CONFIG_ROOT="/workspace/config"

docker_args="-i --security-opt seccomp=unconfined --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host"
docker_args="${docker_args} --volume ${ROOT_DIR}/logs:${LOGS_ROOT}"
docker_args="${docker_args} --volume ${ROOT_DIR}/config:${CONFIG_ROOT}"
docker_args="${docker_args} --volume /dev/shm/dlbs:/workspace/cuda_cache"

${docker_launcher} run ${docker_args} ${docker_img}
