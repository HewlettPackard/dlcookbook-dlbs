#!/bin/bash
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
. ${BENCH_ROOT}/../../../scripts/environment.sh
dlbs=$DLBS_ROOT/python/dlbs/experimenter.py
parser=$DLBS_ROOT/python/dlbs/logparser.py
loglevel=warning
#------------------------------------------------------------------------------#
# Run multi-process multi-GPU inference with synthetic/real data. This can be
# useful for multi-socket NUMA machines to have one process per CPU.
# You may want to run compute_mprocess_throughput.sh to compute adjusted throughput.
#------------------------------------------------------------------------------#
logdir=./logs/mprocess        # Log directory
gpu_list="0,1,2,3;4,5,6,7";   # GPUs for each benchmark process
cpu_list="0-17;18-35"         # CPUs (cores) for each benchmark process
gpus_to_use="0,1,2,3"         # These GPUs will be used by processes after setting visible GPUs

rm -rf ${logdir}
mkdir -p ${logdir}

gpu_list=(${gpu_list//;/ });
cpu_list=(${cpu_list//;/ });
nprocesses=${#gpu_list[@]}
for i in "${!gpus[@]}"
do
    gpus=${gpu_list[$i]}
    cpus=${cpu_list[$i]}
    launcher="CUDA_VISIBLE_DEVICES=${gpus} DLBS_TENSORRT_SYNCH_BENCHMARKS=$i,$nprocesses,dlbs_ipc numactl --localalloc --physcpubind=${cpus}"

    python $dlbs run \
           --log-level=$loglevel\
           -Pruntime.launcher=\"${launcher}\"\
           -Pexp.docker_args='"--rm --ipc=host --privileged"'\
           -Ptensorrt.num_prefetchers=3\
           -Pexp.data_dir='"/path/to/dataset/or/empty"'\
           -Ptensorrt.data_name='"tensors1_OR_tensors4_OR_images_OR_empty"'\
           -Pexp.dtype='"float16"'\
           -Pexp.gpus=\"${gpus_to_use}\"\
           -Vexp.model='["resnet50"]'\
           -Pexp.replica_batch=128\
           -Pexp.num_warmup_batches=20\
           -Pexp.num_batches=200\
           -Ptensorrt.inference_queue_size=6\
           -Ptensorrt.rank=$i\
           -Pexp.log_file='"${BENCH_ROOT}/logs/real/${exp.model}_${exp.gpus}.log"'\
           -Pexp.phase='"inference"'\
           -Pexp.docker=true\
           -Pexp.docker_image='"hpe/tensorrt:cuda9-cudnn7"'\
           -Pexp.framework='"tensorrt"' &
done
wait

params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title,exp.gpus,results.mgpu_effective_throughput"
python $parser ./logs/real/*.log --output_params ${params}
