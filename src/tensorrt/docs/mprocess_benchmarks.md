# Multi process inference benchmarks
The term multi-process multi-GPU inference benchmark describes scenario where on a multi-socket system with large 
number of GPUs (e.g. 8), due to specific CPU <-> GPU connectivity topology and NUMA effects (as well as, possibly,
multi-NIC connectivity to high performance storage), it may be benefitial to run benchmarks with multiple processes.

This means that instead of having one process using 8 GPUs, we have 2 processes each using 4 GPUs. Each process can
be pinned to its CPU to eliminate NUMA effects. This is quite possible because inference is basically an embarrassingly
parallel workload.

To run multi-process benchmarks, users need to take care about several things.
1. Pin processes to cores and do not over-subscribe .
2. In each process use GPUs that are connected to `your` CPU.
3. When having two sockets in 4:1 configuration i.e. when 4 GPUs are connected to CPU0 and another 4 GPUs are connected
   to CPU1, try to use GPUs from different sockets to avoid PCIe congestion i.e.:
     - When running benchmarks with 2 processes each using 2 GPUs, use GPUs, for instance, 0 and 2 with CPU0 and GPUs
       4 and 6 with GPU1 (this may vary depending on your particular HW topology).
4. Use `numactl` to pin processes and enforce local memory allocation for each benchmark process.
5. If your data is stored in memory, make sure you copy it there taking into account NUMA domains. With two processes,
   make sure process A running on socket 0 reads data from NUMA domain 0 and process B running on socket 1 reads data
   from NUMA domain 1.

In order to synchronize benchmark processes for better throughput estimation, `DLBS_TENSORRT_SYNCH_BENCHMARKS`
environment variable is used. Read documentation for `Environment` class to learn more details on format of this
variable and other requirements.

### Running benchmarks with DLBS
DLBS as of now does not natively support running and synchronizing multiple benchmarks at the same time. To enable this,
we use bash scripts to run multiple processes in parallel and we use environment variables to synchronzie benchmarks.
There is also a python script that computes average throughput. Assuming that we are using configuration presented
[here](sprocess_benchmarks.md), the following bash script can be used as a template to run multi-process benchmarks:
```bash
#!/bin/bash
#------------------------------------------------------------------------------#
# This script will not run anything by default. You need to specify configuration
# manually.
#------------------------------------------------------------------------------#
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../../../scripts/environment.sh

#------------------------------------------------------------------------------#
p=/usr/bin/python
dlbs=$DLBS_ROOT/python/dlbs/experimenter.py              # Script that runs benchmarks
parser=$DLBS_ROOT/python/dlbs/logparser.py               # Script that parses log files
sb=$DLBS_ROOT/python/dlbs/reports/summary_builder.py     # Script that builds reports
config=./config.json                                     # Reuse this JSON config

# Run experiments.
cpu0="0-17"     # Range of physical cores for CPU 0
cpu1="18-35"

# Function that runs in parallel two or more benchmarking process.
# Usage:
#    run MODEL REPLICA_BATCH GPUS VGPUS CPUS
# Example:
#    run "alexnet_owt" "512" "0,1,2,3;4,5,6,7" "0,1,2,3" "0-17;18-35"
#       MODEL: alenet_owt
#       REPLICA_BATCH (per-GPU batch size): 512
#       GPUS: "0,1,2,3;4,5,6,7"
#             Run two benchmark process, one uses 0,1,2,3 GPUs and the second one uses
#             4,5,6,7 GPUs
#       VGPUS: "0,1,2,3"
#             Benchmarking Suite will use CUDA_VISIBLE_DEVICE to limit GPU visibility.
#             Benchmarking processes should use this virtual GPUs to run benchmarks.
#       CPUS: "0-17;18-35"
#             First process will bind to NUMA node 0 (0-17), and the second process will
#             bind to NUMA node 1 (18-35).
function run () {
    [ -f "/dev/shm/dlbs_ipc" ] && rm /dev/shm/dlbs_ipc
    model=$1
    batch=$2
    gpu_list=(${3//;/ })
    vgpus=$4
    cpu_list=(${5//;/ })

    nprocesses=${#gpu_list[@]}
    for i in "${!gpu_list[@]}"; do
        gpus=${gpu_list[$i]}
        cpus=${cpu_list[$i]}
        launcher="CUDA_VISIBLE_DEVICES=${gpus} DLBS_TENSORRT_SYNCH_BENCHMARKS=$i,$nprocesses,dlbs_ipc numactl --localalloc --physcpubind=${cpus} "
        echo "#############$launcher###############"
        logdir="${BENCH_ROOT}/logs/${vgpus//,/.}_${model}_${batch}"
        mkdir -p "$logdir"
        $p $dlbs run --log-level=info --config=${config} -Ptensorrt.rank=$i \
                       -Pexp.docker_args='"--rm --ipc=host --privileged"' \
                       -Vexp.gpus=\"${vgpus}\" -Vexp.model=\"${model}\" -Vexp.replica_batch=$batch \
                       -Pruntime.launcher='"'"${launcher}"'"' \
                       -Vexp.log_file='"'"${logdir}"'/rank_${tensorrt.rank}.log''"' \
                       -Pexp.status='""' \
                       -Ptensorrt.num_prefetchers=8 -Ptensorrt.inference_queue_size=64 -Pexp.num_batches=1000 \
                       -Ptensorrt.docker_image='"sergey/tensorrt:cuda9-cudnn7"' \
                       -Ptensorrt.data_dir='"/some/hpc/storage/imagenet100k/rank${tensorrt.rank}"'&
    done
    wait
    echo "-----------------------------------------------------------------------------------------------------"
    echo "Model: $model, replica batch size: $batch, gpus: $3, logdir: $logdir"
    ${DLBS_ROOT}/tutorials/dlcookbook/tensorrt/compute_mprocess_throughput.sh --logdir "${logdir}" --python "$p"
    echo "-----------------------------------------------------------------------------------------------------"
}

# ---------------------------------------------------------------------------------
#   BEST results achieved with this parameters
#  DO NOT FORGET TO CHANGE NUMBER  OF PREFETCHERS IN THE ABOVE FUNCTION
# 2 processes, num prefetchers=16
#run "alexnet_owt" "1024" "0;4" "0" "${cpu0};${cpu1}"
#run "alexnet_owt" "1024" "0,2;4,6" "0,1" "${cpu0};${cpu1}"
#run "alexnet_owt" "1024" "0,1,2,3;4,5,6,7" "0,1,2,3" "${cpu0};${cpu1}"
#run "alexnet_owt" "512" "0;4" "0" "${cpu0};${cpu1}"
#run "alexnet_owt" "512" "0,2;4,6" "0,1" "${cpu0};${cpu1}"
#run "alexnet_owt" "512" "0,1,2,3;4,5,6,7" "0,1,2,3" "${cpu0};${cpu1}"

#run "resnet50" "256" "0,2;4,6" "0,1" "${cpu0};${cpu1}"
#run "resnet50" "256" "0,1,2,3;4,5,6,7" "0,1,2,3" "${cpu0};${cpu1}"
#run "resnet50" "128" "0,2;4,6" "0,1" "${cpu0};${cpu1}"
#run "resnet50" "128" "0,1,2,3;4,5,6,7" "0,1,2,3" "${cpu0};${cpu1}"

# 4 processes, num_prefetchers=8

#run "alexnet_owt" "1024" "0;2;4;6" "0" "0-17;36-53;18-35;54-71"
#run "alexnet_owt" "1024" "0,1;2,3;4,5;6,7" "0,1" "0-17;36-53;18-35;54-71"
#run "alexnet_owt" "512"  "0;2;4;6" "0" "0-17;36-53;18-35;54-71"
#run "alexnet_owt" "512" "0,1;2,3;4,5;6,7" "0,1" "0-17;36-53;18-35;54-71"
# ---------------------------------------------------------------------------------

# How to interpret results?
#   1. Search for lines similar to this one
#         Model: resnet50, replica batch size: 256, gpus: 0,1,2,3;4,5,6,7
#   2. Below, you'll see something like this:
#         Benchmark 0 (per GPU perf): own throughput 21873.500000, effective throughput 20879.900000
#         Benchmark 1 (per GPU perf): own throughput 21319.300000, effective throughput 21318.400000
#         Adjusted throughput: 41321.163497
#      You can select either adjusted througput as the final throughput or
#      max(adjusted throughput, SUM(effective throughput)).

```
