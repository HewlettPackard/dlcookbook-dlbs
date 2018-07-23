#!/bin/bash

script_folder=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
script_fname=$(basename ${BASH_SOURCE[0]})
. ${script_folder}/../scripts/environment.sh
logparser="$DLBS_ROOT/python/dlbs/logparser.py"
summary_builder="$DLBS_ROOT/python/dlbs/reports/summary_builder.py"

type="weak-scaling"
target="results.time"

help_message="\
Usage: ${script_fname} [OPTION...] DIR\n\
Parse log files recursively located in DIR directory and build DLBS textual reports.\n\
\n\
  --type TYPE          A type of the report to build. One of ['weak-scaling', 'strong-scaling',\n\
                       'exploration']. Look at this page to read about differences between weak\n\
                       and strong scaling: https://developer.hpe.com/blog/scaling-deep-learning-workloads\n\
                       Use exploration report for single CPU/GPU training/inference benchmarks.\n\
                       [default: ${type}]\n\
  --target PARAMETER   A target variable to use in reports. One of ['results.time', 'results.jitter']:\n\
                         - results.time    a processing time of a batch.\n\
                         - results.jitter  jitter, may not be available for all frameworks.\n\
                       [default: ${target}]\n\
The directory must contain milti-GPU benchmarks. Multiple models can be there. But all benchmark log\n\
files must correspond to one particular HW/SW configuration. Do not parse with this tool folders\n\
containing multiple frameworks, or, for instance, benchmarks with various input parameters such as\n\
location of an input dataset. Why is this? Study the following sample output:\n\
-----------------------------------------------------------------------------\n\
Weak scaling report.\n\
Inferences Per Second (IPS, throughput)\n\
Network              Batch      1          2          4          8
InceptionV3          32         338        568        1142       2308
ResNet152            256        322        620        1246       2508
ResNet50             256        693        1328       2668       5391
VGG16                256        386        723        1476       2983
-----------------------------------------------------------------------------\n\
Any two benchmarks that have the same composite key (network title, effective batch size and\n\
list of GPU) are considered to be the same. So, this is the rule to follow:\n\
------------------------------------------------------------------------------\n\
|    Never parse directory that contains more than two benchmarks with the   |
|    same key consisting of neural network title, effective batch size and   |
|    list of GPUs. If you do so, reported results will be wrong.             |
------------------------------------------------------------------------------\n\
\n\
Usage:\n\
  ${script_fname} ./logs\n\
      Parse log files recursively located in ./logs directory and build a weak-scaling report.\n\
  ${script_fname} --type 'strong-scaling' ./logs\n\
      Parse log files recursively located in ./logs directory and build a strong-scaling report.\n\
  ${script_fname} --variable jitter ./logs\n\
      Parse log files recursively located in ./logs directory and build a jitter report."


. $DLBS_ROOT/scripts/parse_options.sh || exit 1;

logdir=$1

tmpfile=$(mktemp -t --dry-run dlbs.reporter.XXXXXXXXXXXXXX.json.gz)

python $logparser $logdir  --recursive --output_file $tmpfile || logfatal "Error parsing log files located at '${logdir}'"
python $summary_builder --summary_file ${tmpfile} --type="${type}" --target_variable="${target}" || logfatal "Error parsing summary file ${tmpfile}"

rm ${tmp_file}
