#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/experimenter.py

action=run
loglevel=warning
#------------------------------------------------------------------------------#
# Example: a minimal working example to run TensorFlow. Run one experiment and
# store results in a file.
# If you run multiple experiments, you really want to make sure that experiment
# log file is different for every experiment.

if true; then
    rm -rf ./tensorflow

    python $script $action --log-level=$loglevel\
                           -Pexp.warmup_iters=10\
                           -Pexp.bench_iters=100\
                           -Pexp.device_batch=16\
                           -Pexp.framework='"tensorflow"'\
                           -Pexp.gpus='0'\
                           -Vexp.model='["alexnet"]'\
                           -Pexp.log_file='"${BENCH_ROOT}/tensorflow/${exp.model}.log"'\
                           -Pexp.env='"docker"'\
                           -Pexp.phase='"training"'\
                           -Ptensorflow.docker.image='"hpe/tensorflow:cuda9-cudnn7"'

    python $DLBS_ROOT/python/dlbs/logparser.py ./tensorflow/*.log --keys exp.framework_id exp.effective_batch results.training_time exp.model_title
fi

#------------------------------------------------------------------------------#
# Example: this one runs tensorflow with several models and several batch sizes
if false; then
    python $script $action --log-level=$loglevel\
                           -Pexp.framework='"tensorflow"'\
                           -Pexp.gpus='0'\
                           -Vexp.env='["docker", "host"]'\
                           -Pexp.log_file='"${BENCH_ROOT}/tensorflow/${exp.env}/${exp.model}_${exp.effective_batch}.log"'\
                           -Vexp.model='["alexnet", "googlenet"]'\
                           -Vexp.device_batch='[2, 4]'\

    python $DLBS_ROOT/python/dlbs/logparser.py ./tensorflow/*.log --keys exp.framework_id exp.effective_batch results.training_time exp.model_title
fi
