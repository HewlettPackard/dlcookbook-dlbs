#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/logparser.py

#------------------------------------------------------------------------------#
# Log parser is the first step to result analysis. Log parser takes log files
# produced by experimenter, parses them and produces json summary. The implmenetation
# of a log parser is quite simple. It just loads each log file and extracts from
# that file valid key-value strings. This then becomes the experiment parameters.
# Every parameter that was computed by the experimenter will be parsed. Also,
# experiment results such as training/inference times will be loaded.
# This tutorial demonstrates different use cases for the log parser.

#------------------------------------------------------------------------------#
# Example 1: Parse one file and print results to a standard output. It should print
# out a whole bunch of parameters.
#python $script ./bvlc_caffe/alexnet_2.log

#------------------------------------------------------------------------------#
# Example 2: If we are intrested only in some of the parameters, we can specify
# them on a command line with --keys command line argument. That's OK if some of
# these parameters are not the log files.
#python $script ./bvlc_caffe/alexnet_2.log --keys "exp.framework_title" "exp.model_title"  "exp.effective_batch" "results.training_time" "results.inference_time"

#------------------------------------------------------------------------------#
# Example 3: It's possible to specify as many log files as you want:
python $script ./bvlc_caffe/*.log --keys "exp.framework_title" "exp.model_title"  "exp.effective_batch" "results.training_time" "results.inference_time"

#------------------------------------------------------------------------------#
# Example 4: It's also possible to specify directory. In case of directory, a
# a switch --recursive can be used to find log files in that directory and all its
# subdirectories
#python $script --log-dir ./bvlc_caffe --recursive --keys "exp.framework_title" "exp.model_title"  "exp.effective_batch" "results.training_time" "results.inference_time"

#------------------------------------------------------------------------------#
# Example 5: Finally, the summary can be written to a file
#python $script --summary-file ./bvlc_caffe/summary.json --log-dir ./bvlc_caffe --recursive\
#               --keys "exp.gpus" "exp.framework_title" "exp.model_title"  "exp.effective_batch" "results.training_time" "results.inference_time" "exp.framework_id"
