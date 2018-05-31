#!/bin/bash
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
. ${BENCH_ROOT}/../../../scripts/environment.sh
logparser=$DLBS_ROOT/python/dlbs/logparser.py
processor=$DLBS_ROOT/src/tensorrt/python/compute_mprocess_throughput.py
#------------------------------------------------------------------------------#
logdir=./logs
tmpfile=./.tmp/results.json
n=50

help_message="\
usage: $0 [OPTION]...\n\
In case if inference benchmark is done with multiple host threads, this script\n\
can be used to compute mean throughput. In general, there are three options to\n\
compute average in case of multi-process benchmarks:\n\
  1. Do nothing special and measure throughput as usual summing up throughput of\n\
     individual processes (groups of inference engines). Issue here is that\n\
     processes can start/stop at different moments and this approach can provide\n\
     too optimistic estimates.\n\
  2. Synch processes in the beginning and end of benchmarking process. This, as\n\
     opposed to Option 1, can provide pessimistic estimates, especially, for high\n\
     throughput models such as AlexnetOWT, where each second basically costs\n\
     thousands of images.\n\
  3. Extract batch times, remove certain number of first/last batch times and\n\
     compute throughput based on that value.\n\
This script computes mean time using approach #3 that is usually used in combination\n\
with #2.\n\
\n\
    --logdir DIR        Directory with log files. [default: $logdir]
    --tmpfile FILE      Temporary file with extracted results. [default: $tmpfile]
    --n N               Number of first/last batches to throw away before computing
                        mean throughput. Totally, 2*N points will be removed from
                        consideration. [default: $n]
"

. $DLBS_ROOT/scripts/parse_options.sh || exit 1;

[ -d "$logdir" ] || logfatal "Input directory ($logdir) does not exist"
[ -f "$tmpfile" ] && rm $tmpfile
mkdir -p $(dirname $tmpfile)

params="exp.replica_batch,results.throughput,results.time_data,results.mgpu_effective_throughput,exp.num_gpus"
python ${logparser} ${logdir} --recursive --output_params ${params} --output_file $tmpfile

[ -f "$tmpfile" ] || logfatal "No file found ($tmpfile): parsing error."
python ${processor} ${tmpfile} ${n}
