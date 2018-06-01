#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
. ${BENCH_ROOT}/../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/experimenter.py
parser=$DLBS_ROOT/python/dlbs/logparser.py
action=run
framework=nvcnn
loglevel=warning
#------------------------------------------------------------------------------#
# For more detailed comments see './bvlc_caffe.sh' script.
#------------------------------------------------------------------------------#
# Example: a minimal working example to run NVCNN TensorFlow. Run one experiment and
# store results in a file.
# If you run multiple experiments, you really want to make sure that experiment
# log file is different for every experiment.
# Now, only support for 'training'
if true; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pnvcnn.use_nccl=true\
                           -Pnvcnn.use_xla=false\
                           -Vexp.dtype='["float16", "float32"]'\
                           -Pexp.num_warmup_batches=100\
                           -Pexp.num_batches=400\
                           -Pexp.replica_batch=512\
                           -Pexp.framework='"nvcnn"'\
                           -Vexp.gpus='["0", "0,1", "0,1,2,3", "0,1,2,3,4,5,6,7"]'\
                           -Vexp.model='["alexnet_owt"]'\
                           -Pexp.log_file='"${BENCH_ROOT}/${exp.framework}/${exp.model}_${exp.dtype}_${exp.effective_batch}.log"'\
                           -Pexp.docker=true\
                           -Pexp.phase='"training"'\
                           -Pnvcnn.docker_image='"nvcr.io/nvidia/tensorflow:18.04-py3"'

    params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title,exp.docker"
    python $parser ./$framework/*.log --output_params ${params}
fi
