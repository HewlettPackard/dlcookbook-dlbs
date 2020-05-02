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

# This file needs to be re-implemented in python - it has become too big for a bash script.

# Source common paths
. ../scripts/environment.sh

prefix=                             # Name prefix for all containers i.e. hpe/tensorflow:cuda8-cudnn7
version=                            # Version or SHA commit to fetch. Must be defined in ./versions
docker="docker"                     # Docker executable.
help_message="Usage: $0 [OPTION]... [IMAGE]...\n\
Build docker IMAGE that is located in one of the subfolders.\n\
If no IMAGEs are specified, list available images. IMAGE format is 'framework/tag' e.g:\n\
    bvlc_caffe/cuda8-cudnn7\n\
    intel_caffe/cpu\n\
    tensorrt/18.11\n\
The 'framework' is a folder in this directory. The 'tag' is a subfolder in a particular\n\
framework's folder.
\n\
Optional arguments:\n\
  --help                   Print his help.\n\
  --docker DOCKER_EXEC     An executable for docker. Most common value is 'docker', but also can be\n\
                           'nvidia-docker'. In certain cases, when current user does not belong\n\
                           to a 'docker' group, it should be 'sudo docker'. Default value is 'docker'.\n\
  --prefix PREFIX          Set image prefix 'prefix/..'. Default is 'hpe' or 'dlbs'. By default,\n\
                           all images have the following name: 'prefix/framework:tag' where\n\
                           framework is a folder in this directory and tag is a subfolder in\n\
                           framework's folder. If prefix is empty, no prefix will be used and\n\
                           image name will be set to 'framework:tag'. Default values for docker\n\
                           images in benchmarking suite assume the prefix exists (dlbs/). If you\n\
                           want to use different prefix, make sure to override image name when\n\
                           running experimenter.\n\
                           ----------------------------------------------------------------------\n\
                           UPDATE:\n\
                           We are moving from deprecated prefix 'hpe' to 'dlbs'.  We will be updating\n\
                           standard configurations to use 'dlbs' prefix. Meanwhile, as we update that,
                           we will also be updating this script, so default prefix may be different for
                           different frameworks. As we update all frameworks, this update message will\n\
                           be removed. Frameworks benchmark backends that support new prefix:\n\
                               - TensorRT\n\
                               - PyTorch\n\
                               - MXNET\n\
                               - Caffe2\n\
                               - Intel Caffe\n\
                           ----------------------------------------------------------------------\n\
  --version COMMIT         If supported by a docker file, framework COMMIT to clone from github.\n\
                           Default value is taken from 'versions' file located in this directory.\n\
                           This is not a specific version like 1.4.0, rather it is a commit tag.\n\
                           All docker files will execute the 'git reset --hard \$version' to \n\
                           use particular project state. Default is set to 'master' in docker files.\n\
                           If user provides this command line argument, this commit will be used\n\
                           for ALL builds (user can provide more than one image to build). So, this\n\
                           may be useful when building one docker image or building docker images for\n\
                           one particular framework.\n\
\
Depending on framework and/or tag different sets of compute capabilities will be used to build\n\
framework for. In general case, with CUDA 9 images Volta GPUs are supported (and others that\n\
include at least Pascal). In some cases, the architecture is autodetected (Caffe2, MXNet). In\n\
other cases, docker files list compute capabilities explicitly (Caffe, TensorFlow).
"

# Parse command line options
. ${DLBS_ROOT}/scripts/parse_options.sh

# If no path specified, print list of supported images
if [[ "$#" -eq 0 ]]; then
    docker_images=$(find . -name 'Dockerfile' -printf '%P\n' | xargs dirname | sed 's/\//:/g')
    echo "$docker_images"
    exit 0;
fi

# If version provided (like 'master') and multiple images are requested,
# use this version for all images. Else, read versions from file.
[[ "${version}XXX" == "XXX" ]] && use_version=false || use_version=true

status=0
for dockerfile_dir in "$@"; do
    # If no version provided, read it from file.
    [[ "${use_version}" == "false" ]] && version=""

    # Firstly, check if user specified Dockerfile instead of a directory
    if [[ "$(basename ${dockerfile_dir})" == "Dockerfile" ]] && [[ -f "${dockerfile_dir}" ]]; then
        dockerfile_dir=$(dirname ${dockerfile_dir})
    fi

    # Verify directory exists
    if ! [[ -d "${dockerfile_dir}" ]]; then
        logfatal "Directory does not exist (\"${dockerfile_dir}\")"
        status=1
        continue
    fi

    # Normalize name
    dockerfile_dir=$(cd ${dockerfile_dir} && pwd)

    # Verify that the name actually points to a nested two level subdirectory
    if [[ "$(pwd)" != "$(cd ${dockerfile_dir}/../.. && pwd)" ]]; then
        echo "Invalid argument ${dockerfile_dir}. It must point to a two level nested subdirectory i.e. 'tensorflow/cuda8-cudnn7'."
        status=1
        continue
    fi

    # Extract framework and tag names
    tag=$(basename ${dockerfile_dir})
    name=$(basename $(dirname ${dockerfile_dir}))

    # If version is not provided, get it from ./versions file.
    # Version must be specified.
    if [[ "${version}XXX" == "XXX" ]]; then
        # First, try to get this docker file specific version, then - general version for this framework
        version=$(get_value_by_key ./versions  "${name}\/${tag}")
        if [[ "${version}XXX" == "XXX" ]]; then
            logwarn "Framework version for docker file  '${name}/${tag}' not found in ./versions (trying general framework version)."
            version=$(get_value_by_key ./versions  "${name}")
            if [[ "${version}XXX" == "XXX" ]]; then
                logwarn "Framework version '${name}' not found in ./versions. Will use default defined in docker file for this framework:tag"
            fi
        fi
    fi

    if [[ "${prefix}XXX" == "XXX" ]]; then
        prefix=dlbs
    fi
    #
    img_name=${prefix}/${name}:${tag}                     # something like hpe/caffe:gpu
    assert_files_exist ${dockerfile_dir}/Dockerfile
    dockerfile_dir=${DLBS_ROOT}/docker/${name}/${tag}     # something like caffe/gpu (dir)
    [[ "${version}XXX" == "XXX" ]] && args="" || args="--build-arg version=${version}"
    [[ -n "${http_proxy}" ]] && args="$args --build-arg http_proxy=${http_proxy}"
    [[ -n "${https_proxy}" ]] && args="$args --build-arg https_proxy=${https_proxy}"

    # If we are building tensorrt docker, we need to copy source of benchmark
    # project to docker context folder
    if [[ "${name}" == "tensorrt" ]]; then
        # Old versions of docker files used external TensorRT packages and base CUDA images
        # because at that time NGC did not exist or did not provide TensorRT images.
        # Starting December 2018, DLBS can use TensorRT images from NGC.
        if [[ ! "$tag" == "18.12" ]]; then
            # One special thing about building TensorRT images is that we need to have
            # a TensorRT package in a docker file folder. The name of that file must be
            # specified in a 'versions' file - so we need verify user has copied this
            # file there.
            if [[ "${version}XXX" == "XXX" ]]; then
                logwarn "Will not build TensorRT ($dockerfile_dir) because name of a package is missing in 'versions' file."
                continue
            fi
            if [[ ! -f "${dockerfile_dir}/${version}" ]]; then
                logwarn "Will not build TensorRT ($dockerfile_dir) because TensorRT package ($dockerfile_dir/$version) not found."
                logwarn "You must copy corresponding package (most likely, *.deb file) into that folder."
                logwarn "You can get it from NVIDIA developer site."
                continue
            fi
        fi
        rm -rf -- ${dockerfile_dir}/tensorrt       # Delete if old version exists
        cp -r ../src/tensorrt  ${dockerfile_dir}   # Copy project
    fi

    if [[ "${name}" == "openvino" ]] && [[ ! ${tag} == "19.09-custom-mkldnn" ]]; then
        # Once base image with OpenVINO becomes available, this will not be required anymore.
        # http://registrationcenter-download.intel.com/akdlm/irc_nas/15792/l_openvino_toolkit_p_2019.2.275.tgz
        # http://registrationcenter-download.intel.com/akdlm/irc_nas/15944/l_openvino_toolkit_p_2019.3.334.tgz
        work_dir=$(pwd)
        cd ${dockerfile_dir}
        # Remove first 6 characters
        openvino_fname="${version:6}"
        if [[ ! -d "./${openvino_fname}" ]]; then
            logwarn "Directory with OpenVINO not found (${openvino_fname})"
            if [[ ! -f "./${openvino_fname}.tgz" ]]; then
                logwarn "Archive with OpenVINO not found (${openvino_fname}.tgz)"
                openvino_url="http://registrationcenter-download.intel.com/akdlm/irc_nas/${version}.tgz"
                wget "${openvino_url}" || logfatal "Error downloading OpenVINO from ${openvino_url}"
            else
                loginfo "Found OpenVINO archive (${openvino_fname}.tgz)"
            fi
            tar -xf l_openvino_toolkit*
        else
            loginfo "Found OpenVINO directory (${openvino_fname})"
        fi
        cd ${work_dir}
    fi

    exec="${docker} build -t $img_name $args $dockerfile_dir"

    loginfo "new docker image build started"
    loginfo "framework version: '${version}' [if provided externally or defined in 'versions' file]"
    loginfo "image name: $img_name"
    loginfo "image location: $dockerfile_dir/Dockerfile"
    loginfo "build args: $args"
    loginfo "exec: $exec"

    ${exec}

    ret_code=$?
    loginfo "docker image $img_name build finished with code $ret_code"
    [[ "$ret_code" -ne "0" ]] && status=${ret_code}

    # If it was TensorRT, clean folder
    [[ "$name" == "tensorrt" ]] && rm -rf -- ${dockerfile_dir}/tensorrt
done
exit ${status}
