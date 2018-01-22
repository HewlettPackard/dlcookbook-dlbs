#!/bin/bash
#------------------------------------------------------------------------------#
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../scripts/environment.sh
#------------------------------------------------------------------------------#
script=$DLBS_ROOT/python/dlbs/experimenter.py  # Script that runs benchmarks
parser=$DLBS_ROOT/python/dlbs/logparser.py     # Script that parses log files
action=run
framework=bvlc_caffe
loglevel=warning
#------------------------------------------------------------------------------#
# To get more information on particular parameter, look into python/dlbs/config folder or run
# from a project root folder: python ./python/dlbs/experimenter.py help --param PARAM_NAME
# where PARAM_NAME is a regular expression defining parameter name, for instance:
# ./python/dlbs/experimenter.py help --param exp.gpus
#------------------------------------------------------------------------------#
# Example: a minimal working example to run BVLC Caffe. Run one experiment and
# store results in a file.
# If you run multiple experiments, you really want to make sure that experiment
# log file is different for every experiment - if it's not the case, the valdiator
# component that's enabled by default will detect this and terminate the script
# without running experiments.
if false; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel                `#Set it to 'info' to track benchmarking progress.`\
                           -Pexp.num_warmup_batches=10          `#Number of iterations that do not contribute to performance measurement.`\
                           -Pexp.num_batches=100                `#Number of benchmarking iterations.`\
                           -Pexp.framework='"bvlc_caffe"'       `#Framework to benchmark.`\
                           -Pexp.docker=true                    `#If true run in container, else use host framework installation.`\
                           -Pexp.gpus='0'                       `#Comma separated list of GPUs to use for one benchmark (data parallel schema).`\
                           -Vexp.model='["alexnet"]'            `#Neural network model.`\
                           -Pexp.replica_batch='"16"'           `#Batch size for one replica. Effective batch size is this value times number of replicas (workers).`\
                           -Vexp.phase='["training"]'           `#Benchmark 'training' or 'inference'.`\
                           -Pexp.log_file='"${BENCH_ROOT}/${caffe.fork}_caffe/${exp.model}.log"' `#Log file - each benchmark must have its own log file.`\
                           -Pcaffe.docker_image='"hpe/bvlc_caffe:cuda9-cudnn7"' `#If exp.docker is truem use this image for BVLC Caffe.`

    # The total number of benchmarking iterations is exp.num_warmup_batches + exp.num_batches. First exp.num_warmup_batches iterations
    # backend is not supposed to use to estimate performance. This warm up is required since frameworks most likely will use first
    # iteration (s) to do memory allocations, tune performance etc. Usually, the very first iteration takes much more time than the
    # subsequent ones. Not all frameworks/phase use this. For instance, Caffe in inference mode does not use warmup.
    # A log file will contain results.time key with average effective batch time in milliseconds computed based on exp.num_batches
    # iterations. Depending on a framework, a log file may also contain results.time_data field wich is an array of times for every
    # iteration (raw data). The length of this array should be exp.num_batches.

    # The boolean exp.docker parameter defines environment for benchmark execution - container or bare metal. If it's true, containers
    # are used. Else host installation is used. Documentation describes how experimenter figures out what docker image it should use
    # or where it should search for a host framework installation.

    # The exp.gpus parameter defines GPUs to use. It's a comma separated list of GPU identifiers that define a data parallel training
    # scheam on a per-node basis (nodes are assumed to be homogenous). If value is '0', use one GPU 0. If value is '"0,1,2,3"' use 4
    # GPU. Each GPU is an independent model replica (data parellel training schema).

    # Alternatively, you can sepcify docker image with bvlc_caffe.docker_image
    # See example below for it. Try running this to get more details:
    #    python ./python/dlbs/experimenter.py help --params ^caffe.docker_image
    python $parser ./$framework/*.log --keys exp.status exp.framework_title exp.effective_batch\
                                             results.time results.throughput exp.model_title
fi
#------------------------------------------------------------------------------#
# Example: this example demonstrates how to use input JSON based configuration files
if true; then
    rm -rf ./$framework

    python $script $action --log-level=$loglevel --config=./configs/bvlc_caffe.json

    python $parser ./$framework/*.log --keys exp.status exp.framework_title exp.effective_batch\
                                         results.time results.throughput exp.model_title
fi
#------------------------------------------------------------------------------#
# Example: this one runs BVLC Caffe with several models and several batch sizes.
# It also enables embedded resource monitor so in log files you can find several
# time series with memory consumption, CPU/GPU utulization etc. The monitor is
# enabled if monitoring frequency (0.1 in this example, in seconds) is > 0.
if false; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.framework='"bvlc_caffe"'\
                           -Vexp.docker=true\
                           -Pexp.gpus='0'\
                           -Pmonitor.frequency=0.1\
                           -Pexp.log_file='"${BENCH_ROOT}/${caffe.fork}_caffe/${exp.model}_${exp.effective_batch}.log"'\
                           -Vexp.model='["alexnet", "googlenet"]'\
                           -Vexp.replica_batch='[2, 4]'\
                           -Pbvlc_caffe.docker_image='"hpe/bvlc_caffe:cuda9-cudnn7"'

    # Setting monitor.frequency to positive value (sampling interval in seconds) enables embedded resource monitor.
    # All results will go into corresponding log files. Read documentation section on what metrics are monitored and
    # how to retrieve this data.

    python $parser ./$framework/*.log --keys exp.status exp.framework_title exp.effective_batch\
                                             results.time results.throughput exp.model_title
fi
#------------------------------------------------------------------------------#
# Example: this example shows how to run bare metal BVLC Caffe with custom library path.
# Make sure you have BVLC Caffe installed. Default installation folder is
# $HOME/projects/bvlc_caffe. User can provide custum path to Caffe's executable with
# 'bvlc_caffe.host_path' parameter that has default value '${HOME}/projects/bvlc_caffe/build/tools'.
# You may also adjust value of parameter 'bvlc_caffe.host_libpath' that basically defines
# LD_LIBRARY_PATH for BVLC Caffe.
# Keep in mind that this is not generally required to provide these parameters when using
# docker containers since I assume that environment is properly configured in docker files.
# You may get error when running this script. If it's something related to CUDNN_STATUS_INTERNAL_ERROR,
# then you need to try to run it with sudo. References:
#    https://devtalk.nvidia.com/default/topic/1024761/cuda-setup-and-installation/cudnn_status_internal_error-when-using-cudnn7-0-with-cuda-8-0/
#    https://stackoverflow.com/questions/47060565/tensorflow-only-works-under-root-after-drivers-update
#    https://devtalk.nvidia.com/default/topic/1025900/cudnn-fails-with-cudnn_status_internal_error-on-mnist-sample-execution/
if false; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.framework='"bvlc_caffe"'\
                           -Vexp.docker=false\
                           -Pexp.log_file='"${BENCH_ROOT}/${caffe.fork}_caffe/${exp.model}_${exp.effective_batch}.log"'\
                           -Vexp.model='["alexnet", "googlenet", "vgg16", "vgg19", "resnet50", "resnet101", "resnet152"]'\
                           -Vexp.replica_batch='[4]'\
                           -Pbvlc_caffe.host_libpath='"/opt/OpenBLAS/lib:/usr/local/cuda/lib64"'
    python $parser ./$framework/*.log --keys exp.status exp.framework_title exp.effective_batch\
                                             results.time results.throughput exp.model_title
fi
