#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../scripts/common.sh
script=$DLBS_ROOT/src/python/dlbs/experimenter.py

#------------------------------------------------------------------------------#
# Example: a minimal working example to run TensorRT. Run one experiment and
# store results in a file.
# If you run multiple experiments, you really want to make sure that experiment
# log file is different for every experiment.
#python $script run --log-level=debug\
#                   -Pexp.framework='"tensorrt"'\
#                   -Pexp.env='"docker"'\
#                   -Pexp.gpus='0'\
#                   -Pexp.phase='"inference"'\
#                   -Pexp.model='"bvlc_alexnet"'\
#                   -Pexp.log_file='"${BENCH_ROOT}/tensorrt/inference.log"'

#------------------------------------------------------------------------------#
# Example: this one runs tensorflow with several models and several batch sizes
#python $script run --log-level=debug\
#                   -Pexp.framework='"caffe2"'\
#                   -Pexp.env='"docker"'\
#                   -Pexp.gpus='0'\
#                   -Pexp.log_file='"${BENCH_ROOT}/caffe2/${exp.model}_${exp.effective_batch}.log"'\
#                   -Vexp.model='["bvlc_alexnet", "bvlc_googlenet"]'\
#                   -Vexp.device_batch='[2, 4]'
