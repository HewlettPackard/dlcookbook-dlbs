#!/bin/bash

loginfo "[STEP 01] Initializing host environment"
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
. ${BENCH_ROOT}/../../../../scripts/environment.sh


loginfo "[STEP 02] Cleaning previous run"
[[ -d ./logs ]] && { rm -rf ./logs || logfatal "Cannot remove results from previous run"; }
mkdir ./logs || logfatal "Cannot create folder for log files"


loginfo "[STEP 03] Pulling docker image"
docker pull nvcr.io/nvidia/tensorflow:18.07-py3 || logfatal "Docker images cannot be pulled"


loginfo "[STEP 04] Running 8 TensorFlow benchmarks (2 models and 4 batch sizes)"
python ${experimenter} run -Pexp.framework='"nvtfcnn"'\
                           -Vexp.model='["resnet50", "alexnet_owt"]'\
                           -Vexp.replica_batch='[4, 8, 16, 32]'\
                           -Pexp.log_file='"./logs/${exp.model}_${exp.replica_batch}.log"'\
                           -Pexp.docker_image='"nvcr.io/nvidia/tensorflow:18.07-py3"'


loginfo "[STEP 05] Parsing summary and building benchmark report"
python ${benchdata} report ./logs --report regular


loginfo "[STEP 07] All done, bye bye"
exit 0
