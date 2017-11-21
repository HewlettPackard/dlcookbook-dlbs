#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/experimenter.py

#------------------------------------------------------------------------------#
# Example: a minimal working example to run INTEL Caffe. Run one experiment and
# store results in a file.
# If you run multiple experiments, you really want to make sure that experiment
# log file is different for every experiment.
#python $script run --log-level=debug\
#                   -Pexp.framework='"nvidia_caffe"'\
#                   -Pexp.env='"docker"'\
#                   -Pexp.gpus='0'\
#                   -Pexp.model='"bvlc_alexnet"'\
#                   -Pexp.log_file='"${BENCH_ROOT}/${caffe.fork}_caffe/training.log"'

#------------------------------------------------------------------------------#
# Example: this one runs tensorflow with several models and several batch sizes
# We do not need any additional libs for this version of Caffe.
python $script run --log-level=debug\
                   -Pexp.framework='"intel_caffe"'\
                   -Vexp.env='["docker", "host"]'\
                   -Pexp.gpus='""'\
                   -Pexp.log_file='"${BENCH_ROOT}/${caffe.fork}_caffe/${exp.env}/${exp.model}_${exp.effective_batch}.log"'\
                   -Vexp.model='["alexnet", "googlenet"]'\
                   -Vexp.device_batch='[2, 4]'
