#!/bin/bash
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

# This script automates building process of docker images located in subdirectories
# of this directory. Each subdirectory corresponds to one deep learning framework.
# Each framework directory contains subdirectories which names define image tags.
# Those tags should be meaningful, specifying, for instance, target device (cpu/gpu)
# or specific enironment settings like cuda or cudnn versions.

# Get GPU name
# nvidia-smi --query-gpu=name --format=csv,noheader -i 0

# Usage: sudo build.sh framework/tag

# Default compute capabilities for various NVIDIA architectures.
# They are used if user specifies fermi/kepler/maxwell/pascal/volta
# as value for 'cuda_arch_bin'
fermi_cc="20,21(20)"
kepler_cc="30,35,37"
maxwell_cc="50,52"
pascal_cc="60,61"
volta_cc="70"

# Source common paths
. ../scripts/environment.sh

prefix=hpe                          # Name prefix for all containers i.e. hpe/tensorflow:cuda8-cudnn7
version=                            # Version or SHA commit to fetch. Must be defined in ./versions
cuda_arch_bin="30,35,50,60,61"      # Defaul list of compute capabilities to use [if supported by a framework]
cuda_arch_ptx="EMPTY"               # PTX values (virtual architectures). Default is same as 'cuda_arch_bin'.
os=ubuntu                           # Reserved for future use
cuda=8                              # Reserved for future use
cudnn=6                             # Reserved for future use
help_message="Usage: $0 [OPTION]... [IMAGE]...\n\
Build docker IMAGE that is located in one of the subfolders.\n\
If no IMAGE specified, list available images.\n\
IMAGE format is 'framework/tag' e.g:\n\
  bvlc_caffe/cuda8-cudnn7\n\
  intel_caffe/cpu\n\
  tensorrt/cuda8-cudnn6\n\

\
Optional arguments:\n\
  --help             Print his help.\n\
  --prefix           Set image prefix 'prefix/..'.\n\
                     Default is '${prefix}'.\n\
  --version          If supported by a docker file,\n\
                     framework version to clone from github.\n\
                     Default is taken from 'versions' file\n\
                     located in this directory.\n\
  --cuda_arch_bin    CUDA architecture to build framework for.\n\
                     Same as cmake arg CUDA_ARCH_BIN.\n\
                     Default is '${cuda_arch_bin}'. Can be a GPU arch:\n\
                       fermi     '${fermi_cc}'\n\
                       kepler    '${kepler_cc}'\n\
                       maxwell   '${maxwell_cc}'\n\
                       pascal    '${pascal_cc}'\n\
                       volta     '${volta_cc}'\n\
  --cuda_arch_ptx    CUDA PTX intermidiate architecture.\n\
                     Same as cmake arg CUDA_ARCH_PTX.\n\
                     Default is same as 'cuda_arch_bin'.
"
#Reserved for future use :\n\
#  -o,  --os           Container OS. Default is '${os}'.\n\
#  -c,  --cuda         CUDA version. Influences base docker image.\n\
#                      Default is '${cuda}'.\n\
#  -n,  --cudnn        CUDNN version. Influences base docker image.\n\
#                      Default is '${cudnn}'.
#"

# Parse command line options
. $DLBS_ROOT/scripts/parse_options.sh

# https://en.wikipedia.org/wiki/CUDA
# https://docs.opencv.org/2.4/modules/gpu/doc/introduction.html
if [ "${_bin}" == "fermi" ]; then
    cuda_arch_bin=${fermi_cc}
elif [ "${cuda_arch_bin}" == "kepler" ]; then
    cuda_arch_bin=${kepler_cc}
elif [ "${cuda_arch_bin}" == "maxwell" ]; then
    cuda_arch_bin=${maxwell_cc}
elif [ "${cuda_arch_bin}" == "pascal" ]; then
    cuda_arch_bin=${pascal_cc}
elif [ "${cuda_arch_bin}" == "volta" ]; then
    cuda_arch_bin=${volta_cc}
else
    # This must be a sequence of compute capabilities
    # TODO: add regular epression check and range
    :
fi

if [ "${cuda_arch_ptx}" == "EMPTY" ]; then
    cuda_arch_ptx=${cuda_arch_bin}
fi

# If no path specified, print list of supported images
if [ "$#" -eq 0 ]; then
    docker_images=$(find . -name 'Dockerfile' -printf '%P\n' | xargs dirname | sed 's/\//:/g')
    echo "$docker_images"
    exit 0;
fi

# If version provided (like 'master') and multiple images are requested,
# use this version for all images. Else, read versions from file.
[ "${version}XXX" == "XXX" ] && use_version=false || use_version=true

status=0
for dockerfile_dir in "$@"; do
    # If no version provided, read it from file.
    [ "${use_version}" == "false" ] && version=""

    # Firstly, check if user specified Dockerfile instead of a directory
    if [ "$(basename $dockerfile_dir)" == "Dockerfile" ] && [ -f "$dockerfile_dir" ]; then
        dockerfile_dir=$(dirname $dockerfile_dir)
    fi

    # Verify directory exists
    if ! [ -d "$dockerfile_dir" ]; then
        logfatal "Directory does not exist (\"$dockerfile_dir\")"
        status=1
        continue
    fi

    # Normalize name
    dockerfile_dir=$(cd $dockerfile_dir && pwd)

    # Verify that the name actually points to a nested two level subdirectory
    if [ "$(pwd)" != "$(cd $dockerfile_dir/../.. && pwd)" ]; then
        echo "Invalid argument ${dockerfile_dir}. It must point to a two level nested subdirectory i.e. 'tensorflow/cuda8-cudnn7'."
        status=1
        continue
    fi

    # Extract framework and tag names
    tag=$(basename $dockerfile_dir)
    name=$(basename $(dirname $dockerfile_dir))

    # If version is not provided, get it from ./versions file.
    # Version must be specified.
    if [ "${version}XXX" == "XXX" ]; then
        # First, try to get this docker file specific version, then - general version for this framework
        version=$(get_value_by_key ./versions  "${name}\/${tag}")
        if [ "${version}XXX" == "XXX" ]; then
            logwarn "Framework version for docker file  '${name}/${tag}' not found in ./versions (trying general framework version)."
            version=$(get_value_by_key ./versions  "${name}")
            if [ "${version}XXX" == "XXX" ]; then
                logwarn "Framework version '${name}' not found in ./versions. Will use default defined in docker file for this framework:tag"
            fi
        fi
    fi

    img_name=$prefix/$name:$tag                     # something like hpe/caffe:gpu
    assert_files_exist $dockerfile_dir/Dockerfile
    dockerfile_dir=$DLBS_ROOT/docker/$name/$tag     # something like caffe/gpu (dir)
    [ "${version}XXX" == "XXX" ] && args="" || args="--build-arg version=${version}"
    args="${args} --build-arg cuda_arch_bin=${cuda_arch_bin} --build-arg cuda_arch_ptx=${cuda_arch_ptx}"
    [ -n "$http_proxy" ] && args="$args --build-arg http_proxy=$http_proxy"
    [ -n "$https_proxy" ] && args="$args --build-arg https_proxy=$https_proxy"

    # If we are building tensorrt docker, we need to copy source of benchmark
    # project to docker context folder
    if [ "$name" == "tensorrt" ]; then
        rm -rf -- $dockerfile_dir/tensorrt       # Delete if old version exists
        cp -r ../src/tensorrt  $dockerfile_dir   # Copy project
    fi


    exec="docker build -t $img_name $args $dockerfile_dir"

    loginfo "new docker image build started"
    loginfo "framework version: '${version}' [if provided externally or defined in 'versions' file]"
    loginfo "cuda BIN architecture: ${cuda_arch_bin} [if applicable]"
    loginfo "cuda PTX architecture: ${cuda_arch_ptx} [if applicable]"
    loginfo "image name: $img_name"
    loginfo "image location: $dockerfile_dir/Dockerfile"
    loginfo "build args: $args"
    loginfo "exec: $exec"

    ${exec}
    ret_code=$?
    loginfo "docker image $img_name build finished with code $ret_code"
    [ "$ret_code" -ne "0" ] && status=$ret_code

    # If it was TensorRT, clean folder
    [ "$name" == "tensorrt" ] && rm -rf -- $dockerfile_dir/tensorrt
done
exit $status
