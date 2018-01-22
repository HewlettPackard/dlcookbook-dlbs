#!/bin/bash
#------------------------------------------------------------------------------#
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../scripts/environment.sh
#------------------------------------------------------------------------------#
script=$DLBS_ROOT/python/dlbs/experimenter.py  # Script that runs benchmarks
parser=$DLBS_ROOT/python/dlbs/logparser.py     # Script that parses log files
#action=validate
action=run
loglevel=warning

#------------------------------------------------------------------------------#
# This example demonstrates one way to control what benchmarks should be performed. Problem statement - run benchmarks with different
# models, frameworks and batch sizes and containers from NIDIA GPU cloud. What should be taken into account - for large models we cannot
# fit large batch sizes into GPU memory. The other thing is that we may want to specify exact batch sizes for different models.
# Baseline solution - run all models with all batch sizes. Some experiments will fail. Then, use only those results that are required.
if true; then
    python $script $action\
        --log-level=$loglevel\
        -Pmonitor.frequency=0\
        -Pexp.status='"disabled"'\
        -Vexp.framework='["nvidia_caffe","tensorflow","caffe2","mxnet"]'\
        -Vexp.gpus='["0","0,1","0,1,2,3"]'\
        -Vexp.model='["alexnet", "googlenet", "resnet50", "vgg16", "resnet152"]'\
        -Vexp.replica_batch='[8,16,32,64,128,256,512,1024,2048]'\
        -Pexp.log_file='"${BENCH_ROOT}/${exp.framework}/$(\"${exp.gpus}\".replace(\",\",\".\"))$_${exp.model}_${exp.effective_batch}.log"'\
        -Pexp.docker=true\
        -Pnvidia_caffe.docker_image='"nvcr.io/nvidia/caffe:17.12"'\
        -Ptensorflow.docker_image='"nvcr.io/nvidia/tensorflow:17.12"'\
        -Pcaffe2.docker_image='"nvcr.io/nvidia/caffe2:17.12"'\
        -Pmxnet.docker_image='"nvcr.io/nvidia/mxnet:17.12"'\
        -E'{"condition":{"exp.model": "alexnet", "exp.replica_batch":[64,128,256,512]},"parameters":{"exp.status":""}}'\
        -E'{"condition":{"exp.model": "googlenet", "exp.replica_batch":[64,128,256,512]},"parameters":{"exp.status":""}}'\
        -E'{"condition":{"exp.model": "resnet50", "exp.replica_batch":[32,64,128]},"parameters":{"exp.status":""}}'\
        -E'{"condition":{"exp.model": "vgg16", "exp.replica_batch":[32,64,128]},"parameters":{"exp.status":""}}'\
        -E'{"condition":{"exp.model": "resnet152", "exp.replica_batch":[16,32,64]},"parameters":{"exp.status":""}}'

    #python $parser ./$framework/*.log --keys exp.status exp.framework_title exp.effective_batch\
    #                                         results.time results.throughput exp.model_title\
    #                                         exp.gpus exp.docker_image

    # This is what happens here. We enumerate all possible configuration that include those that we are intrested in. Look at exp.framework,
    # exp.gpus and exp.replica_batch parameters. We then disable all these configurations (exp.status). We then specify docker images that we
    # want to use (-Pexp.docker=true enables docker containers and *.docker_image parameters specify exact images to use)

    # What is the complete set of experiments:
    #   - Four frameworks - BVLC Caffe, TensorFlow, Caffe2 and mxnet.
    #   - Four GPU configurations that define single/multiple (distributed) GPU experiments.
    #   - Five neural network models - AlexNet, GoogleNet, ResNet50, VGG16 and resnet152
    #   - Nine replica batch sizes - 8, 16, 32, 64, 128, 256, 512, 1024 and 2048

    # Then, we define conditions that enable certain configurations that we are interested in.
    #    If model is AlexNet, enable these bath sizes (64,128,256,512) for all frameworks/GPU configurations
    #    If model is GoogleNet, enable these bath sizes (64,128,256,512) for all frameworks/GPU configurations
    #    If model is ResNet50, enable these bath sizes (32,64,128) for all frameworks/GPU configurations
    # and so on ...

    # Originally, we have this number of benchmarks: 4 * 4 * 5 * 9 = 720
    # They are all disabled, we enable these number of experiments: (4 + 4 + 3 + 3 + 3) * 4 * 4 = 272
fi
#------------------------------------------------------------------------------#
# Example: almost same configuration as above but with config defined in JSON file
if false; then
    python $script $action --log-level=$loglevel --config=./configs/advanced.json
fi
#------------------------------------------------------------------------------#
