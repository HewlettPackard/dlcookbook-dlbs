#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
. ${BENCH_ROOT}/../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/experimenter.py
parser=$DLBS_ROOT/python/dlbs/logparser.py
action=run
framework=tensorrt
loglevel=warning
#------------------------------------------------------------------------------#
# For more detailed comments see './bvlc_caffe.sh' script.
#------------------------------------------------------------------------------#
# Example: a minimal working example to run TensorRT. Run one experiment and
# store results in a file.
# If you run multiple experiments, you really want to make sure that experiment
# log file is different for every experiment.
# This example runs in a container.
if true; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.framework='"tensorrt"'\
                           -Pexp.docker=true\
                           -Pexp.gpus='0'\
                           -Pexp.phase='"inference"'\
                           -Vexp.model='["resnet18"]'\
                           -Pexp.log_file='"${BENCH_ROOT}/tensorrt/${exp.model}.log"'
    params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title"
    python $parser ./$framework/*.log --output-params ${params}
fi
#------------------------------------------------------------------------------#
# Example: same experiment as above but runs in a host OS. I must run this as a root
# sudo .... Do not know why for now, related thread:
# https://devtalk.nvidia.com/default/topic/1024906/tensorrt-3-0-run-mnist-sample-error-assertion-engine-failed/
if false; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.framework='"tensorrt"' -Pexp.docker=false\
                           -Pexp.gpus='0' -Pexp.phase='"inference"'\
                           -Vexp.model='["resnet18"]'\
                           -Pexp.log_file='"${BENCH_ROOT}/tensorrt/${exp.model}.log"'
    params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title"
    python $parser ./$framework/*.log --output-params ${params}
fi
#------------------------------------------------------------------------------#
# Example: this one runs TensorRT with several models and several batch sizes
if false; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                   -Pexp.framework='"tensorrt"'\
                   -Pexp.docker=true\
                   -Pexp.gpus='0'\
                   -Pexp.log_file='"${BENCH_ROOT}/tensorrt/${exp.model}_${exp.effective_batch}.log"'\
                   -Vexp.model='["alexnet", "googlenet", "deep_mnist", "eng_acoustic_model", "resnet50", "resnet101", "resnet152", "vgg16", "vgg19"]'\
                   -Vexp.replica_batch='[2, 4]'\
                   -Pexp.phase='"inference"'\
                   -Pexp.num_warmup_batches=1\
                   -Pexp.num_batches=1
    params="exp.framework_title,exp.effective_batch,results.time,results.total_time,exp.model_title"
    python $parser ./$framework/*.log --output-params ${params}
fi
