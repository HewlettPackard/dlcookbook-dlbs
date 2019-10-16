#!/bin/bash

export ROOT_DIR=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
. ${ROOT_DIR}/../../../../scripts/environment.sh


builder=${DLBS_ROOT}/python/dlbs/mlbox/mlbox_builder.py
python ${builder} --config "./config.json"\
                  --hashtag "7c5ca5a6dfa4e2f7b8b4d81c60bd8be343dabd30"\
                  --work_dir "./mlbox"\
                  --base_image "nvcr.io/nvidia/tensorflow:18.07-py3"\
                  --docker_image "dlbs/ngc-tf:18.07-mlbox"\
                  --docker_launcher "nvidia-docker"
