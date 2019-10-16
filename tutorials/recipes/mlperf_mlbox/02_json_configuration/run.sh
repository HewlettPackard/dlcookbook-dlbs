#!/bin/bash

# The config.json defines a benchmark configuration for a 4-way GPU server. It runs 123 benchmarks in total. Depending
# on your GPUs, some will fail due to Out-Of-Memory errors caused by large replica batch sizes.

# The following variables are referenced in the configuration and must be defined (for logging purposes, may be empty):
DLBS_USER="FirstName SecondName"
DLBS_USER_EMAIL=""
NODE_ID=""
NODE_TITLE=""
GPU_TITLE=""


loginfo "[STEP 01] Initializing host environment"
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
. ${BENCH_ROOT}/../../../../scripts/environment.sh


# Actually, I am not going to clean folders - users must do that on their own (do not want accidentally remove all
# benchmark data)
#loginfo "[STEP 02] Cleaning previous run"
#[[ -d ./logs ]] && { rm -rf ./logs || logfatal "Cannot remove results from previous run"; }
#mkdir ./logs || logfatal "Cannot create folder for log files"
[[ ! -d ./logs ]] && { mkdir ./logs || logfatal "Cannot create folder for log files"; }


loginfo "[STEP 03] Pulling docker image"
docker pull nvcr.io/nvidia/tensorflow:19.09-py3 || logfatal "Docker images cannot be pulled"


loginfo "[STEP 04] Running N TensorFlow benchmarks"
python ${experimenter} run --config ./config.json


loginfo "[STEP 05] Parsing log files and serializing benchmark summary in a large JSON file"
results=./results/benchmarks.json.gz
python $benchdata parse ${BENCH_ROOT}/logs --output ${results}


loginfo "[STEP 06] Building various reports"
build_training_reports() {
    local tool out_dir prec_report img_report data
    data=$1
    create_dirs ./results/training/${data}/weak ./results/training/${data}/precision/ ./results/training/${data}/containers
    tool="python ${benchdata} report ${results} --report"
    out_dir=./results/training/${data}
    for ver in 19.09; do
        docker_img='"exp.docker_image": "nvcr.io/nvidia/tensorflow:'"${ver}"'-py3"'
        # WEAK scaling
        for dtype in float32 float16; do
            ${tool} weak --select '{'"${docker_img}"', "exp.dtype": "'"${dtype}"'", "exp.data": "'${data}'"}' > ${out_dir}/weak/nvtfcnn_${ver}_${dtype}.txt
        done

        # PRECISION report
        prec_report='{"inputs": ["exp.model_title", "exp.num_gpus", "exp.replica_batch"], "output_cols": ["float32", "float16"], "output": "exp.dtype", "report_speedup": true}'
        for ngpus in 1 2 4; do
            ${tool} "${prec_report}" --select '{'"${docker_img}"', "exp.num_gpus": '"${ngpus}"', "exp.data": "'${data}'"}' > ${out_dir}/precision/gpus${ngpus}.txt
        done
    done
}
build_training_reports synthetic


loginfo "[STEP 07] All done, bye bye"
exit 0
