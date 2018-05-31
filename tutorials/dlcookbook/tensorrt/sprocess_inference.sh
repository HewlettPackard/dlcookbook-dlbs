#!/bin/bash
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
. ${BENCH_ROOT}/../../../scripts/environment.sh
dlbs=$DLBS_ROOT/python/dlbs/experimenter.py
parser=$DLBS_ROOT/python/dlbs/logparser.py
loglevel=warning
#------------------------------------------------------------------------------#
# To bind process to a specific CPU, use something like this:
#   -Pruntime.launcher='"numactl --localalloc --physcpubind=0-17 "'
#------------------------------------------------------------------------------#
rm -rf ./logs/sprocess
python $dlbs run \
       --log-level=$loglevel\
       -Pruntime.launcher='" "'\
       -Vexp.gpus='["0"]'\
       -Vexp.model='["alexnet_owt"]'\
       -Pexp.replica_batch=512\
       -Pexp.num_warmup_batches=100\
       -Pexp.num_batches=500\
       -Ptensorrt.inference_queue_size=4\
       -Ptensorrt.data_dir='"/path/to/dataset/or/empty"'\
       -Ptensorrt.data_name='"tensors1"'\
       -Pexp.log_file='"${BENCH_ROOT}/logs/sprocess/${exp.model}_${exp.num_gpus}.log"'\
       -Pexp.phase='"inference"'\
       -Pexp.docker=true\
       -Pexp.docker_image='"hpe/tensorrt:cuda9-cudnn7"'\
       -Pexp.framework='"tensorrt"'

params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title"
python $parser ./logs/synthetic/*.log --output_params ${params}
