#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
. ${BENCH_ROOT}/../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/experimenter.py
parser=$DLBS_ROOT/python/dlbs/logparser.py
action=run
framework=nvidia_caffe
loglevel=warning
#------------------------------------------------------------------------------#
# For more detailed comments see './bvlc_caffe.sh' script.
#------------------------------------------------------------------------------#
# Example: a minimal working example to run NVIDIA Caffe. Run one experiment and
# store results in a file.
# If you run multiple experiments, you really want to make sure that experiment
# log file is different for every experiment.
if true; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.num_warmup_batches=10\
                           -Pexp.num_batches=100\
                           -Pexp.framework='"nvidia_caffe"'\
                           -Pexp.docker=true\
                           -Pexp.gpus='0'\
                           -Vexp.model='["alexnet"]'\
                           -Pexp.replica_batch=16\
                           -Vexp.phase='["training"]'\
                           -Pexp.log_file='"${BENCH_ROOT}/${caffe.fork}_caffe/${exp.model}.log"'\
                           -Pcaffe.docker.image='"hpe/nvidia_caffe:cuda9-cudnn7"'
    python $parser ./$framework/*.log --keys exp.status exp.framework_title exp.effective_batch\
                                             results.time results.throughput exp.model_title
fi
#------------------------------------------------------------------------------#
# Example: this one runs NVIDIA Caffe with several models and several batch sizes
if false; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.framework='"nvidia_caffe"'\
                           -Vexp.docker=true\
                           -Pexp.gpus='0'\
                           -Pexp.log_file='"${BENCH_ROOT}/${caffe.fork}_caffe/${exp.model}_${exp.effective_batch}.log"'\
                           -Vexp.model='["alexnet", "googlenet"]'\
                           -Vexp.replica_batch='[2, 4]'\
                           -Pcaffe.docker.image='"hpe/nvidia_caffe:cuda9-cudnn7"'
    python $parser ./$framework/*.log --keys exp.status exp.framework_title exp.effective_batch\
                                             results.time results.throughput exp.model_title
fi

#------------------------------------------------------------------------------#
# Example: this example shows how to run bare metal NVIDIA Caffe with custom library path
if false; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.framework='"nvidia_caffe"'\
                           -Vexp.docker=false\
                           -Pexp.gpus='"0,1"'\
                           -Pexp.log_file='"${BENCH_ROOT}/${caffe.fork}_caffe/${exp.model}_${exp.effective_batch}.log"'\
                           -Vexp.model='["alexnet"]'\
                           -Vexp.replica_batch='[16]'\
                           -Pnvidia_caffe.host_libpath='"/opt/OpenBLAS/lib:/opt/hdf5-1.10.1/lib:/opt/cudnn-7.0.3/lib64:/opt/nccl/lib:/opt/boost-1.63.0/lib:/usr/local/lib"'\
                           -Pcaffe.data_dir='"/fdata/imagenet-data/lmdb/ilsvrc12_train_lmdb/"'\
                           -Pcaffe.data_mean_file='"/fdata/imagenet-data/lmdb/imagenet_mean.binaryproto"'
    python $parser ./$framework/*.log --keys exp.status exp.framework_title exp.effective_batch\
                                             results.time results.throughput exp.model_title
fi
#-Vexp.model='["alexnet", "googlenet", "vgg16", "vgg19", "resnet50", "resnet101", "resnet152"]'\
#------------------------------------------------------------------------------#
# Run networks with different precision.
# On Voltas with CUDA 9 and cuDNN7 this script can use tensor cores no matter
# what value is assigned to exp.use_tensor_core.
if false; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.num_warmup_batches=10\
                           -Pexp.num_batches=100\
                           -Pexp.framework='"nvidia_caffe"'\
                           -Pexp.docker=true\
                           -Pexp.gpus='0'\
                           -Pexp.replica_batch=16\
                           -Vexp.model='["alexnet"]'\
                           -Vexp.phase='["training"]'\
                           -Vnvidia_caffe.precision='["float32", "float16", "mixed"]'\
                           -Pexp.log_file='"${BENCH_ROOT}/${caffe.fork}_caffe/${nvidia_caffe.precision}/${exp.model}.log"'\
                           -Pnvidia_caffe.docker_image='"hpe/nvidia_caffe:cuda9-cudnn7"'
    python $parser --log-dir ./$framework\
                   --recursive\
                   --keys exp.status exp.framework_title exp.effective_batch\
                          results.time results.throughput exp.model_title\
                          nvidia_caffe.precision
fi
