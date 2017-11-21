#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/experimenter.py

action=run
#------------------------------------------------------------------------------#
# Example: a minimal working example to run TensorRT. Run one experiment and
# store results in a file.
# If you run multiple experiments, you really want to make sure that experiment
# log file is different for every experiment.
# This example runs in a host OS. You need to have TensorRT and build a TensorRT
# benchmarker in $DLBS_ROOT/src/tensorrt
# I need to run it as a root. Need to investigate further.
if false; then
    rm -rf ./tensorrt

    python $script $action --log-level=debug\
                           -Pexp.framework='"tensorrt"'\
                           -Pexp.env='"host"'\
                           -Pexp.gpus='0'\
                           -Pexp.phase='"inference"'\
                           -Pexp.model='"deep_mnist"'\
                           -Pexp.log_file='"${BENCH_ROOT}/tensorrt/deep_mnist.log"'

    python $DLBS_ROOT/python/dlbs/logparser.py ./tensorrt/*.log --keys exp.framework_id exp.effective_batch results.inference_time results.total_time exp.model_title
fi

if true; then
    rm -rf ./tensorrt

    python $script $action --log-level=debug\
                           -Pexp.framework='"tensorrt"'\
                           -Pexp.env='"docker"'\
                           -Pexp.gpus='0'\
                           -Pexp.phase='"inference"'\
                           -Pexp.model='"deep_mnist"'\
                           -Pexp.log_file='"${BENCH_ROOT}/tensorrt/deep_mnist.log"'\
                           -Ptensorrt.docker.image='"hpe/tensorrt:cuda8-cudnn6"'

    python $DLBS_ROOT/python/dlbs/logparser.py ./tensorrt/*.log --keys exp.framework_id exp.effective_batch results.inference_time results.total_time exp.model_title
fi
#------------------------------------------------------------------------------#
# Example: this one runs TensorRT with several models and several batch sizes
if false; then
    rm -rf ./tensorrt
    python $script $action --log-level=debug\
                   -Pexp.framework='"tensorrt"'\
                   -Pexp.env='"host"'\
                   -Pexp.gpus='0'\
                   -Pexp.log_file='"${BENCH_ROOT}/tensorrt/${exp.model}_${exp.effective_batch}.log"'\
                   -Vexp.model='["bvlc_alexnet", "bvlc_googlenet", "deep_mnist", "eng_acoustic_model", "resnet50", "resnet101", "resnet152", "vgg16", "vgg19"]'\
                   -Vexp.device_batch='[2, 4]'\
                   -Pexp.phase='"inference"'\
                   -Pexp.warmup_iters=1\
                   -Pexp.bench_iters=1\

    python $DLBS_ROOT/python/dlbs/logparser.py ./tensorrt/*.log --keys exp.framework_id exp.effective_batch results.inference_time results.total_time exp.model_title
fi
