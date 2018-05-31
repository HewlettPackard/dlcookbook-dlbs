#!/bin/bash
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
# Warmup system and/or run storage/network benchmarks by streaming images into
# host memory and measuring the throughput. This will provide an intuition on what
# maximal throughput can be achieved taking into account only ability of a system
# to stream data from storage to host memory.
#------------------------------------------------------------------------------#
data_dir=~/data/tensors1
dtype=uchar
img_size=227
batch_size=512
num_prefetchers=2
num_preallocated_batches=4
num_warmup_batches=10
num_batches=50

docker_img="hpe/tensorrt:cuda9-cudnn7"
cpus=""

help_message="\
usage: $0 [OPTION]...\n\
Warmup system and/or run storage/network benchmarks by streaming images into\n\
host memory and measuring the throughput. This will provide an intuition on what\n\
maximal throughput can be achieved taking into account only ability of a system\n\
to stream data from storage to host memory.\n\
\n\
    --data_dir DIR                 This datset needs to be created with images2tensors\n\
                                   tool. [default: $data_dir]
    --dtype float|uchar            Data type for image arrays in dataset:\n\
                                       'float' - 4 bytes\n\
                                       'uchar' (unsigned char) - 1 byte\n\
                                   [default: $dtype]
    --img_size SIZE                Size of input images [3, img_size, img_size] in dataset.\n\
                                   [default: $img_size]
    --batch_size SIZE              Read data in batches using this batch size.\n\
                                   [default: $batch_size]
    --num_prefetchers N            Number of parallel threads reading data from dataset.\n\
                                   [default: $num_prefetchers]
    --num_preallocated_batches N   Number of preallocated batches. Readers do not allocate memory,\n\
                                   rather, they use preallocated memory from pool of batches.\n\
                                   [default: $num_preallocated_batches]
    --num_warmup_batches N         Number of warmup iterations. [default: $num_warmup_batches]\n\
    --num_batches N                Number of benchmark iterations. [default: $num_batches]\n\
    --docker_img NAME              Docker image [default: $docker_img]\n\
    --cpus CPU_RANGE               Pin process to these range of CPUs with numactl and\n\
                                   enforce local memory allocation policy. For instance,\n\
                                   on a two socket NUMA machine with 18 cores per CPU,\n\
                                   setting --cpus 0-17 will effectively pin process to\n\
                                   socket #0. [default: do not pin process]\n\
"

. ${BENCH_ROOT}/../../../scripts/environment.sh
. $DLBS_ROOT/scripts/parse_options.sh || exit 1;


docker_args="-ti --rm --volume=${data_dir}:/mnt/dataset"
[ ! "$cpus" == "" ] && \
                 { binder="numactl --localalloc --physcpubind=$cpus"; \
                   loginfo "Will bind benchmark process: $binder"; \
                   docker_args="$docker_args --privileged"; \
                 } || binder=""

# We will run this command in a container. Do not change this line.
exec="benchmark_tensor_dataset --data_dir=/mnt/dataset --batch_size=${batch_size}"
exec="${exec} --img_size=${img_size} --num_prefetchers=${num_prefetchers} --dtype=${dtype}"
exec="${exec} --prefetch_pool_size=${num_preallocated_batches} --num_warmup_batches=${num_warmup_batches}"
exec="${exec} --num_batches=${num_batches}"

docker run ${docker_args} ${docker_img} /bin/bash -c "${binder} ${exec}"
exit 0
