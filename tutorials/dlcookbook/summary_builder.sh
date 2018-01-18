#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../scripts/environment.sh

#------------------------------------------------------------------------------#
# Summary builde can build three types of reports:
#   exploration      Suitable for a one GPU/CPU systems or for inference phase.
#   weak-scaling     Tables outlining weak scaling results of conducted experiments.
#   strong-scaling   Tables outlining weak strong results of conducted experiments
# Weak/strong scaling required multi-GPU experiments to be performed. They are suitable
# for multi-GPU systems with 2, 4 and 8 GPUs.
script=$DLBS_ROOT/python/dlbs/reports/summary_builder.py

#------------------------------------------------------------------------------#
# Example 1: Build a simple exploration report for framework 'bvlc_caffe'. Every
# record in input json file must contain the following fields to be succesfully used:
#   1. 'exp.framework' with 'bvlc_caffe' value else record is skipped
#   2. 'results.time' with some numerical value
#   3. 'exp.model_title' with title of a neural network model
#   4. 'exp.gpus' with list of GPUs used in experiment
#   5. 'exp.effective_batch' with effective batch of an experiment.
# Such summary file can be generated with a log parser script.
if true; then
python $script --summary-file ./bvlc_caffe/summary.json\
               --type='exploration'\
               --target-variable='results.time'\
               --query='{"exp.framework": "bvlc_caffe"}'
fi
