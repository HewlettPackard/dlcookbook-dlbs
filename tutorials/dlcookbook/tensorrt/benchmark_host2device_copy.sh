#!/bin/bash
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache

gpu=0
size=10
pinned_mem=true
num_warmup_batches=10
num_batches=100
docker_img="hpe/tensorrt:cuda9-cudnn7"
cpus=""

help_message="\
usage: $0 [OPTION]...\n\
Benchmark host to device memory transfer. This will provide an intuition on what\n\
maximal throughput can be achieved taking into account only ability of a system\n\
to stream data from host to device memory.\n\
For instance, let's consider that this benchmark says that you can stream data\n\
at 10.5 GB/sec (not very good taking into account 16 PCIe-3 lanes, but anyway).\n\
Assuming that data is transfered as floating point arrays of length\n\
BatchSize*3*Wight*Height where Wight=Height=227 (AlexNet), then you can expect\n\
inference throughput at most 10.5*1024^3/(3*227^2*4)~18000 images/sec. This,\n\
however, will never be achieved due to various factors: (1) compute intensive\n\
neural network model which cannot be processed at this rate and (2) software\n\
implementation of an inference function that does not optimally overlap all\n\
comptue/copy operations.\n\
\n\
    --gpu GPU_ID               GPU index to use. [default: $gpu]\n\
    --size DATA_SIZE           Size of a data chunk in MegaBytes. During inference\n\
                               benchmarks, data is transfered as arrays of shape\n\
                               [BatchSize, 3, Wight, Height] of 'float' data type.\n\
                               These are typical sizes for AlexNetOWT where\n\
                               Width = Height = 227:\n\
                               Batch size (images):  32  64  128  256  512  1024\n\
                               Batch size (MB):      19  38   75  151  302   604\n\
                               [default: $size]\n\
    --pinned_mem true|false    Allocate buffer in host pinned memory. [default: $pinned_mem]\n\
    --num_warmup_batches N     Number of warmup iterations. [default: $num_warmup_batches]\n\
    --num_batches N            Number of benchmark iterations. [default: $num_batches]\n\
    --docker_img NAME          Docker image [default: $docker_img]\n\
    --cpus CPU_RANGE           Pin process to these range of CPUs with numactl and\n\
                               enforce local memory allocation policy. For instance,\n\
                               on a two socket NUMA machine with 18 cores per CPU,\n\
                               setting --cpus 0-17 will effectively pin process to\n\
                               socket #0. [default: do not pin process]\n\
"

. ${BENCH_ROOT}/../../../scripts/environment.sh
. $DLBS_ROOT/scripts/parse_options.sh || exit 1;


docker_args="-ti --rm"
[ "$pinned_mem" == "true" ] && pinned="--pinned" || pinned=""
[ ! "$cpus" == "" ] && \
                 { binder="numactl --localalloc --physcpubind=$cpus"; \
                   loginfo "Will bind benchmark process: $binder"; \
                   docker_args="$docker_args --privileged"; \
                 } || binder=""
# We will run this command in a container. Do not change this line.
exec="benchmark_host2device_copy --gpu=$gpu --size=$size $pinned --num_warmup_batches=${num_warmup_batches} --num_batches=${num_batches}"
nvidia-docker run ${docker_args}  ${docker_img} /bin/bash -c "${binder} ${exec}"
exit 0
