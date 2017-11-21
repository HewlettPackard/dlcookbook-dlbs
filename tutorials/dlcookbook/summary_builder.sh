#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/reports/summary_builder.py

#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Example 1: Parse one file and print results to a standard output. It should print
# out a whole bunch of parameters.
#python $script --summary-file ./bvlc_caffe/summary.json\
#               --type='strong-scaling'\
#               --target-variable='results.training_time'\
#               --query='{"exp.framework_id": "bvlc_caffee"}'
