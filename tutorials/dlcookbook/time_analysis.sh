#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/reports/time_analysis.py

#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
# Example 1: Parse one file and print results to a standard output. It should print
# out a whole bunch of parameters.
#python $script --recursive --log-dir ./mxnet
python $script --recursive --log-dir ./caffe2-max-perf
