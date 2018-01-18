#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../scripts/environment.sh
#------------------------------------------------------------------------------#
# This script builds 2D scatter plot with X being batch update and Y being
# time of that batch. It also plots smoothed versions. Useful to study drift in
# batch times. This will work only for frameworks that output time series of
# batch values - Caffe2, MXNet and TensorRT.
# User will need GUI to see results.
script=$DLBS_ROOT/python/dlbs/reports/time_analysis.py

#------------------------------------------------------------------------------#
# Example 1: Plot chart for one experiment
if true; then
    python $script --log-file ./mxnet/alexnet_float32_16.log
fi
#------------------------------------------------------------------------------#
# Example 1: Plot chart for multiple experiments
if false; then
    python $script --log-dir ./mxnet
fi
