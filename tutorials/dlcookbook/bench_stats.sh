#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../scripts/environment.sh

#------------------------------------------------------------------------------#
# Bench stats script scans raw log files and computes simple statistics on
# performed experiments. It gives a very high level overview on what experiments
# have beem performed / succedded / failed.
# Run this script after running other scripts that produce log files. By default,
# it's assumed that this folder contains 'bvlc_caffe' folder with one or more files,
# possible, located in sub-directories.
script=$DLBS_ROOT/python/dlbs/reports/bench_stats.py

#------------------------------------------------------------------------------#
# Example: Parse logging directory recursively and print results to a standard
# output. It should print out a whole bunch of parameters.
python $script --log_dir ./bvlc_caffe --recursive
