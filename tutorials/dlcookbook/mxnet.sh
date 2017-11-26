#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/experimenter.py

action=run
loglevel=warning
#------------------------------------------------------------------------------#
# Example: a minimal working example to run MxNet. Run one experiment and
# store results in a file.
# If you run multiple experiments, you really want to make sure that experiment
# log file is different for every experiment.

if true; then
    rm -rf ./mxnet

    python $script $action --log-level=$loglevel\
                           -Pexp.framework='"mxnet"'\
                           -Pexp.gpus='0'\
                           -Pexp.env='"docker"'\
                           -Vexp.device_batch='[16]'\
                           -Pexp.warmup_iters=10\
                           -Pexp.bench_iters=100\
                           -Pmxnet.cudnn_autotune='"false"'\
                           -Vexp.model='["alexnet"]'\
                           -Pexp.phase='"training"'\
                           -Pexp.log_file='"${BENCH_ROOT}/mxnet/${exp.model}_${exp.effective_batch}.log"'\
                           -Pmxnet.docker.image='"hpe/mxnet:cuda9-cudnn7"'

    python $DLBS_ROOT/python/dlbs/logparser.py ./mxnet/*.log --keys exp.framework_id exp.effective_batch results.training_time exp.model_title
fi

#------------------------------------------------------------------------------#
# Example: this one runs tensorflow with several models and several batch sizes
if false; then
    rm -rf ./mxnet

    python $script $action --log-level=$loglevel\
                           -Pexp.framework='"mxnet"'\
                           -Pexp.gpus='0'\
                           -Vexp.env='["docker", "host"]'\
                           -Pexp.log_file='"${BENCH_ROOT}/mxnet/${exp.env}/${exp.model}_${exp.effective_batch}.log"'\
                           -Vexp.model='["alexnet", "googlenet"]'\
                           -Vexp.device_batch='[2, 4]'\
                           -Vmxnet.host.libpath='"/opt/OpenBLAS/lib"'

    python $DLBS_ROOT/python/dlbs/logparser.py ./mxnet/*.log --keys exp.framework_id exp.effective_batch results.training_time exp.model_title
fi
