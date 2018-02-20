#!/bin/bash
#------------------------------------------------------------------------------#
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../scripts/environment.sh
#------------------------------------------------------------------------------#
script=$DLBS_ROOT/python/dlbs/experimenter.py
parser=$DLBS_ROOT/python/dlbs/logparser.py
action=run
framework=caffe2
loglevel=warning
#------------------------------------------------------------------------------#
# For more detailed comments see './bvlc_caffe.sh' script.
#------------------------------------------------------------------------------#
# Example: a minimal working example to run Caffe2. Run one experiment and
# store results in a file.
# If you run multiple experiments, you really want to make sure that experiment
# log file is different for every experiment.
if true; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.num_warmup_batches=10\
                           -Pexp.num_batches=100\
                           -Pexp.framework='"caffe2"'\
                           -Pexp.gpus='0'\
                           -Pexp.docker=true\
                           -Pexp.phase='"training"'\
                           -Vexp.replica_batch='[16]'\
                           -Vexp.model='["alexnet"]'\
                           -Pexp.log_file='"${BENCH_ROOT}/caffe2/${exp.model}_${exp.effective_batch}.log"'\
                           -Pcaffe2.docker.image='"hpe/caffe2:cuda9-cudnn7"'
    params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title,exp.docker_image"
    python $parser ./$framework/*.log --output_params ${params}
fi
#------------------------------------------------------------------------------#
# Example: this one runs Caffe2 in host OS. By default, it's assumed caffe2
# is located in $HOME/projects/caffe2. Users can override the default location
# of a Caffe2 executable with parameter 'caffe2.host_python_path' (default value is
# ${HOME}/projects/caffe2/build). Also, it may be required to provide paths where
# dependencies are located. This can be done with caffe2.host_libpath parameter
# (default value is ${HOME}/projects/caffe2/build/caffe2).
if false; then
    rm -rf ./$framework
    python $script run --log-level=$loglevel\
                   -Pexp.framework='"caffe2"'\
                   -Pexp.gpus='"0"'\
                   -Vexp.docker=false\
                   -Pexp.log_file='"${BENCH_ROOT}/caffe2/${exp.model}_${exp.effective_batch}.log"'\
                   -Vexp.model='["alexnet"]'\
                   -Vexp.replica_batch='[8]'\
                   -Pexp.num_warmup_batches=100\
                   -Pexp.num_batches=100
    params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title,exp.docker_image"
    python $parser ./$framework/*.log --output_params ${params}
fi
#------------------------------------------------------------------------------#
# Compare FP32/FP16. FP16 benchmark may fail if your system does not support it.
if false; then
    rm -rf ./$framework
    python $script $action --log-level=$loglevel\
                           -Pexp.num_warmup_batches=10\
                           -Pexp.num_batches=100\
                           -Pexp.framework='"caffe2"'\
                           -Pexp.gpus='0'\
                           -Pexp.docker=true\
                           -Pexp.phase='"training"'\
                           -Pexp.replica_batch=16\
                           -Vexp.model='["alexnet"]'\
                           -Vexp.dtype='["float32", "float16"]'\
                           -Pexp.log_file='"${BENCH_ROOT}/caffe2/${exp.model}_${exp.dtype}_${exp.effective_batch}.log"'\
                           -Pcaffe2.docker.image='"hpe/caffe2:cuda9-cudnn7"'
    params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title,exp.docker_image"
    python $parser ./$framework/*.log --output_params ${params}
fi
