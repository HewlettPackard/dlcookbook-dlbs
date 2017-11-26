#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/experimenter.py

action=run
loglevel=warning
#------------------------------------------------------------------------------#
# Example: a minimal working example to run Caffe2. Run one experiment and
# store results in a file.
# If you run multiple experiments, you really want to make sure that experiment
# log file is different for every experiment.
if true; then
    rm -rf ./caffe2

    python $script $action --log-level=$loglevel\
                           -Pexp.warmup_iters=10\
                           -Pexp.bench_iters=100\
                           -Pexp.framework='"caffe2"'\
                           -Pexp.gpus='0'\
                           -Pexp.env='"docker"'\
                           -Pexp.phase='"training"'\
                           -Vexp.device_batch='[16]'\
                           -Vexp.model='["alexnet"]'\
                           -Pexp.log_file='"${BENCH_ROOT}/caffe2/${exp.model}_${exp.effective_batch}.log"'\
                           -Pcaffe2.docker.image='"hpe/caffe2:cuda9-cudnn7"'

    python $DLBS_ROOT/python/dlbs/logparser.py ./caffe2/*.log --keys exp.framework_id exp.effective_batch results.training_time exp.model_title
fi

#------------------------------------------------------------------------------#
# Example: this one runs Caffe2 in host OS

if false; then
    rm -rf ./caffe2

    python $script run --log-level=$loglevel\
                   -Pexp.docker.launcher='"nvidia-docker"'\
                   -Pexp.framework='"caffe2"'\
                   -Pexp.gpus='"0"'\
                   -Pexp.device='"gpu"'\
                   -Vexp.env='["host"]'\
                   -Pexp.log_file='"${BENCH_ROOT}/caffe2/data/${exp.env}/${exp.model}_${exp.effective_batch}.log"'\
                   -Vexp.model='["alexnet"]'\
                   -Vexp.device_batch='[8]'\
                   -Pexp.warmup_iters=100\
                   -Pexp.bench_iters=100

    python $DLBS_ROOT/python/dlbs/logparser.py ./caffe2/*.log --keys exp.framework_id exp.effective_batch results.training_time exp.model_title
fi