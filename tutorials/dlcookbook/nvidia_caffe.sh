#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/experimenter.py

action=run
framework=nvidia_caffe
loglevel=warning
#------------------------------------------------------------------------------#
# Example: a minimal working example to run NVIDIA Caffe. Run one experiment and
# store results in a file.
# If you run multiple experiments, you really want to make sure that experiment
# log file is different for every experiment.

if true; then
    rm -rf ./$framework

    python $script $action --log-level=$loglevel\
                           -Pexp.warmup_iters=10\
                           -Pexp.bench_iters=100\
                           -Pexp.framework='"nvidia_caffe"'\
                           -Pexp.env='"docker"'\
                           -Pexp.gpus='0'\
                           -Vexp.model='["alexnet"]'\
                           -Pexp.device_batch=16\
                           -Vexp.phase='["training"]'\
                           -Pexp.log_file='"${BENCH_ROOT}/${caffe.fork}_caffe/${exp.model}.log"'\
                           -Pcaffe.docker.image='"hpe/nvidia_caffe:cuda9-cudnn7"'

    python $DLBS_ROOT/python/dlbs/logparser.py ./$framework/*.log --keys exp.framework_id exp.effective_batch results.inference_time results.training_time exp.model_title
fi
#------------------------------------------------------------------------------#
# Example: this one runs NVIDIA Caffe with several models and several batch sizes

if false; then
    rm -rf ./$framework

    python $script $action --log-level=$loglevel\
                           -Pexp.framework='"nvidia_caffe"'\
                           -Vexp.env='["docker"]'\
                           -Pexp.gpus='0'\
                           -Pexp.log_file='"${BENCH_ROOT}/${caffe.fork}_caffe/${exp.model}_${exp.effective_batch}.log"'\
                           -Vexp.model='["alexnet", "googlenet"]'\
                           -Vexp.device_batch='[2, 4]'\
                           -Pcaffe.docker.image='"hpe/nvidia_caffe:cuda9-cudnn7"'

    python $DLBS_ROOT/python/dlbs/logparser.py ./$framework/*.log --keys exp.framework_id exp.effective_batch results.training_time exp.model_title
fi

#------------------------------------------------------------------------------#
# Example: this example shows how to run bare metal BVLC Caffe with custom library path

if false; then
    rm -rf ./$framework

    python $script $action --log-level=info\
                           -Pexp.framework='"nvidia_caffe"'\
                           -Vexp.env='["host"]'\
                           -Pexp.gpus='0'\
                           -Pexp.log_file='"${BENCH_ROOT}/${caffe.fork}_caffe/${exp.model}_${exp.effective_batch}.log"'\
                           -Vexp.model='["alexnet", "googlenet", "vgg16", "vgg19", "resnet50", "resnet101", "resnet152"]'\
                           -Vexp.device_batch='[4]'\
                           -Pnvidia_caffe.host.libpath='"/opt/OpenBLAS/lib:/opt/hdf5-1.10.1/lib:/opt/cudnn-7.0.3/lib64:/opt/nccl1/lib"'

    python $DLBS_ROOT/python/dlbs/logparser.py ./$framework/*.log --keys exp.framework_id exp.effective_batch results.training_time exp.model_title
fi


