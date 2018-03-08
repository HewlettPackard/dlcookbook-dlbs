#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
. ${BENCH_ROOT}/../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/experimenter.py
parser=$DLBS_ROOT/python/dlbs/logparser.py
action=run
framework=pytorch
loglevel=warning
#------------------------------------------------------------------------------#
# For more detailed comments see './bvlc_caffe.sh' script.
#------------------------------------------------------------------------------#
# Example: a minimal working example to run PyTorch. Run one experiment and
# store results in a file. If you run multiple experiments, you really want to
# make sure that experiment log file is different for every experiment.
if false; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel \
                           -Pexp.framework='"pytorch"' \
                           -Ppytorch.cudnn_benchmark=false \
                           -Ppytorch.cudnn_fastest=true \
                           -Pexp.gpus='"0"' \
                           -Pexp.docker=true \
                           -Pexp.replica_batch='8' \
                           -Pexp.dtype='"float32"' \
                           -Pexp.model='"resnet50"' \
                           -Pexp.phase='"training"' \
                           -Pexp.log_file='"${BENCH_ROOT}/pytorch/${exp.model}_${exp.effective_batch}.log"' \
                           -Pexp.docker_args='"--rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864"'
    params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title,exp.docker_image"
    python $parser ./$framework/*.log --output_params ${params}
fi

#------------------------------------------------------------------------------#
# Example: this one runs PyTorch with one model and several GPUs (weak scaling). 
if true; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.framework='"pytorch"'\
                           -Vexp.gpus='["0", "0,1", "0,1,2,3"]'\
                           -Pexp.docker=true\
                           -Pexp.log_file='"${BENCH_ROOT}/pytorch/${exp.model}_${exp.effective_batch}_${exp.num_gpus}.log"'\
                           -Vexp.model='["googlenet"]'\
                           -Vexp.replica_batch='[32]'\
                           -Pexp.docker_args='"--rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864"'
    params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title,exp.docker_image"
    python $parser ./$framework/*.log --output_params ${params}                           
fi

#------------------------------------------------------------------------------#
# NEXT STEPS
#  1. Run benchmarks with half precision (-Pexp.dtype='"float16"')
#  2. Run benchmarks with real data (-Ppytorch.data_dir='"PATH"')
#     Two data backends are supported - 'caffe_lmdb' and 'image_folder'. Default
#       is caffe_lmdb (-Ppytorch.data_backend='"caffe_lmdb"')
#     With real data, you may want to increase number of data loader threads
#      (-Pnum_loader_threads=N), default is 4.
