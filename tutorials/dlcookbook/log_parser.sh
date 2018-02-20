#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/logparser.py

#------------------------------------------------------------------------------#
# Log parser is the first step to do result analysis. Log parser takes log files
# produced by experimenter, parses them and produces json summary. The implmenetation
# of a log parser is quite simple. It just loads each log file and extracts from
# that file valid key-value strings. This then becomes the experiment parameters.
# Every parameter that was computed by the experimenter will be parsed. Also,
# experiment results such as training/inference times will be loaded.
# This tutorial demonstrates different use cases for the log parser.

#------------------------------------------------------------------------------#
# Example: Parse one file and print results to a standard output. It should print
# out a whole bunch of parameters.
if true; then
    python $script ./bvlc_caffe/alexnet*.log
fi

#------------------------------------------------------------------------------#
# Example: If we are intrested only in some of the parameters, we can specify
# them on a command line with ''--output_params' command line argument. That's OK
# if some of these parameters are not the log files.
if false; then
    params="exp.framework_title,exp.model_title,exp.effective_batch,results.time"
    python $script ./bvlc_caffe/alexnet*.log --output_params ${params}
fi

#------------------------------------------------------------------------------#
# Example: It's possible to specify as many log files as you want:
if false; then
    params="exp.framework_title,exp.model_title,exp.effective_batch,results.time"
    python $script ./bvlc_caffe/*.log --output_params ${params}
fi
#------------------------------------------------------------------------------#
# Example: It's also possible to specify directory. In case of directory, a
# a switch --recursive can be used to find log files in that directory and all its
# subdirectories
if false; then
    params="exp.framework_title,exp.model_title,exp.effective_batch,results.time"
    python $script ./bvlc_caffe --recursive --output_params ${params}
fi

#------------------------------------------------------------------------------#
# Example: The summary can be written to a file. A content of the output
# file can be used to build summary reports (see 'summary_builder.sh' file).
if false; then
    params="exp.framework_title,exp.model_title,exp.effective_batch,results.time"
    python $script ./bvlc_caffe --recursive \
                                --output_file ./bvlc_caffe.json \
                                --output_params ${params}
fi

#------------------------------------------------------------------------------#
# Example: The summary can be written to a gzipped file.
if false; then
    params="exp.framework_title,exp.model_title,exp.effective_batch,results.time"
    python $script ./bvlc_caffe --recursive \
                                --output_file ./bvlc_caffe.json.gz \
                                --output_params ${params}
fi

#------------------------------------------------------------------------------#
# Example: It's possible to selectively filter benchmarks setting constraints
# on parameter value
if false; then
    params="exp.framework_title,exp.model_title,exp.effective_batch,results.time"
    python $script ./bvlc_caffe --recursive \
                                --output_params ${params} \
                                --filter_query='{"exp.model": "alexnet"}'
fi

#------------------------------------------------------------------------------#
# Example: It's also possible to add new parameters. Values for new parameters
# can reference existing parameters
if false; then
    #params="exp.framework_title,exp.model_title,exp.effective_batch,results.time"
    params="exp.my_summary"
    python $script ./bvlc_caffe \
      --recursive \
      --output_params ${params} \
      -P'{"exp.my_summary":"${exp.framework_title}/${exp.model_title}/${exp.effective_batch} -> ${results.time}"}'
fi
