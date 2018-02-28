#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
. ${BENCH_ROOT}/../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/experimenter.py
parser=$DLBS_ROOT/python/dlbs/logparser.py
action=run
framework=mxnet
loglevel=warning
#------------------------------------------------------------------------------#
# For more detailed comments see './bvlc_caffe.sh' script.
#------------------------------------------------------------------------------#
# Example: a minimal working example to run MxNet. Run one experiment and
# store results in a file. If you run multiple experiments, you really want to
# make sure that experiment log file is different for every experiment. This
# experiment also uses embedded resource monitor that tracks system resources
# consumed by a benchmarking process such as CPU/GPU utilization, memory etc.
if true; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.framework='"mxnet"'\
                           -Pexp.gpus='0'\
                           -Pexp.docker=true\
                           -Pmonitor.frequency=0.1\
                           -Vexp.replica_batch='[16]'\
                           -Pexp.num_warmup_batches=10\
                           -Pexp.num_batches=10\
                           -Pmxnet.cudnn_autotune='"false"'\
                           -Vexp.model='["alexnet"]'\
                           -Pexp.phase='"training"'\
                           -Pexp.log_file='"${BENCH_ROOT}/mxnet/${exp.model}_${exp.effective_batch}.log"'\
                           -Pmxnet.docker_image='"hpe/mxnet:cuda9-cudnn7"'
    params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title,exp.docker_image"
    python $parser ./$framework/*.log --output_params ${params}
fi

#------------------------------------------------------------------------------#
# Example: this one runs tensorflow with several models and several batch sizes. It
# will use sevearl models, batch sizes and will use containers and local host installation
# to run experiments.
if false; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.framework='"mxnet"'\
                           -Pexp.gpus='0'\
                           -Pmonitor.frequency=0\
                           -Vexp.docker='[true, false]'\
                           -Pexp.log_file='"${BENCH_ROOT}/mxnet/$(\"docker\" if ${exp.docker} else \"host\")$/${exp.model}_${exp.effective_batch}.log"'\
                           -Vexp.model='["alexnet", "googlenet"]'\
                           -Vexp.replica_batch='[2, 4]'\
                           -Vmxnet.host_libpath='"/opt/OpenBLAS/lib"'
    params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title,exp.docker"
    python $parser ./$framework/*.log --output_params ${params}
fi

#------------------------------------------------------------------------------#
# Compare performance with float32 and float16 precision.
if false; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.framework='"mxnet"'\
                           -Pexp.gpus='0'\
                           -Pexp.docker=true\
                           -Pexp.replica_batch=16\
                           -Pexp.num_warmup_batches=10\
                           -Pexp.num_batches=100\
                           -Pmxnet.cudnn_autotune=true\
                           -Pexp.use_tensor_core=true\
                           -Pexp.model='"alexnet"'\
                           -Vexp.dtype='["float32", "float16"]'\
                           -Pexp.phase='"training"'\
                           -Pexp.log_file='"${BENCH_ROOT}/mxnet/${exp.model}_${exp.dtype}_${exp.effective_batch}.log"'\
                           -Pmxnet.docker_image='"hpe/mxnet:cuda9-cudnn7"'
    params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title,exp.docker_image,exp.dtype"
    python $parser ./$framework/*.log --output_params ${params}
fi
