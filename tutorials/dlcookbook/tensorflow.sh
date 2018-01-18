#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
. ${BENCH_ROOT}/../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/experimenter.py
parser=$DLBS_ROOT/python/dlbs/logparser.py
action=run
framework=tensorflow
loglevel=warning
#------------------------------------------------------------------------------#
# For more detailed comments see './bvlc_caffe.sh' script.
#------------------------------------------------------------------------------#
# Example: a minimal working example to run TensorFlow. Run one experiment and
# store results in a file.
# If you run multiple experiments, you really want to make sure that experiment
# log file is different for every experiment.
if true; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.num_warmup_batches=10\
                           -Pexp.num_batches=100\
                           -Pexp.replica_batch=16\
                           -Pexp.framework='"tensorflow"'\
                           -Pexp.gpus='0'\
                           -Vexp.model='["alexnet"]'\
                           -Pexp.log_file='"${BENCH_ROOT}/tensorflow/${exp.model}.log"'\
                           -Pexp.docker=true\
                           -Pexp.phase='"training"'\
                           -Ptensorflow.docker_image='"hpe/tensorflow:cuda9-cudnn7"'
    python $parser ./$framework/*.log --keys exp.status exp.framework_title exp.effective_batch\
                                             results.time results.throughput exp.model_title\
                                             exp.docker_image
fi

#------------------------------------------------------------------------------#
# Example: this one runs tensorflow with several models and several batch sizes
if false; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.framework='"tensorflow"'\
                           -Pexp.gpus='0'\
                           -Vexp.docker='[true, false]'\
                           -Pexp.log_file='"${BENCH_ROOT}/tensorflow/$(\"docker\" if ${exp.docker} else \"host\")$/${exp.model}_${exp.effective_batch}.log"'\
                           -Vexp.model='["alexnet", "googlenet"]'\
                           -Vexp.replica_batch='[2, 4]'\
    python $parser ./$framework/*.log --keys exp.status exp.framework_title exp.effective_batch\
                                             results.time results.throughput exp.model_title\
                                             exp.docker
fi
