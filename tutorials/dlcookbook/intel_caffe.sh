#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../scripts/environment.sh
#------------------------------------------------------------------------------#
script=$DLBS_ROOT/python/dlbs/experimenter.py
parser=$DLBS_ROOT/python/dlbs/logparser.py
action=run
framework=intel_caffe
loglevel=warning
#------------------------------------------------------------------------------#
# For more detailed comments see './bvlc_caffe.sh' script.
#------------------------------------------------------------------------------#
# Example: a minimal working example to run Intel Caffe. Run one experiment and
# store results in a file.
# If you run multiple experiments, you really want to make sure that experiment
# log file is different for every experiment.
if true; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.num_warmup_batches=10\
                           -Pexp.num_batches=100\
                           -Pexp.framework='"intel_caffe"'\
                           -Pexp.docker=true\
                           -Pexp.replica_batch=16\
                           -Pexp.gpus='""'\
                           -Vexp.model='["alexnet"]'\
                           -Vexp.phase='["training"]'\
                           -Pexp.log_file='"${BENCH_ROOT}/${caffe.fork}_caffe/${exp.model}.log"'
    params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title,exp.phase"
    python $parser ./$framework/*.log --output_params ${params}
fi
#------------------------------------------------------------------------------#
# Example: this one runs Intel Caffe with several models and several batch sizes.
# It also runs Intel Caffe in container and host OS, so, make sure you have both.
if false; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.framework='"intel_caffe"'\
                           -Vexp.docker='[true, false]'\
                           -Pexp.gpus='""'\
                           -Pexp.log_file='"${BENCH_ROOT}/${caffe.fork}_caffe/$(\"docker\" if ${exp.docker} else \"host\")$/${exp.model}_${exp.effective_batch}.log"'\
                           -Vexp.model='["alexnet", "googlenet"]'\
                           -Vexp.replica_batch='[2, 4]'
    params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title,exp.docker_image,exp.phase"
    python $parser ./$framework/ --recursive --output_params ${params}
fi
