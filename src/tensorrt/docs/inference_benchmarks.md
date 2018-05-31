# Inference Benchmarks

In addition to command line parameters accepted by the inference benchmark tool
and exposed via DLBS, environment variables can be used to tune inference benchmark
process:

##### __DLBS_TENSORRT_NO_PINNED_MEMORY__
By default, memory is allocated as pinned memory that speeds up CPU-GPU transfers.
This may, however, degrade performance if a server does not have sufficient RAM.
If it is the case, assign the variable a value 1:
```bash
export DLBS_TENSORRT_NO_PINNED_MEMORY=1
```

##### __DLBS_TENSORRT_SYNCH_BENCHMARKS__
The inference benchmark tool can use multiple GPUs to run inference benchmarks.
For instance, if you have an 8-GPU box, one process can use all 8 GPUs. Sometimes, depending on a HW configuration, it can be more beneficial in terms of performance
to run multiple processes with smaller number of GPUs. For instance, if you have
a two socket NUMA server in a 4 by 1 configuration (4 GPUs connected to one CPU and
another 4 GPUs connected to another CPU), you can run two benchmark processes on
these two CPUs pinning these processes with numactl, something like:
```bash
numactl --localalloc --physcpubind=0-17 ...
```
In this particular case, the question is how to measure the overall performance
across all benchmarks. There are several ways, one of which is to synch processes
at the start and at the end and then use the total time to compute aggregated
throughput. The DLBS_TENSORRT_SYNCH_BENCHMARKS variable is used exactly for this
purpose. The format of this variable is the following: `my_rank,num_processes,name`. Where `num_processes` is the total number of processes to synchronize, `my_rank`
is the process identifier and `name` is a semaphore name (semaphores are used
for cross-process synchronization, think about it as a file name in /dev/shm).
If you run benchmarks in docker containers, do not forget to run containers with
`--ipc=host`. For instance:
```bash
# Process 1
export DLBS_TENSORRT_SYNCH_BENCHMARKS=0,2,dlbs_ipc
# Process 2
export DLBS_TENSORRT_SYNCH_BENCHMARKS=1,2,dlbs_ipc
```

##### DLBS_TENSORRT_INFERENCE_IMPL_VER
The inference benchmark tool provides two implementations of an inference function.
Version 1, or legacy, is the original implementation that uses default CUDA stream
to copy data and do inference. The high level logic is the following: (1) copy data to GPU, (2) do inference and (3) copy results back to host memory. To enable this
implementation, export the following variale:
```bash
export DLBS_TENSORRT_INFERENCE_IMPL_VER=1
```
This implementation is sort of OK when input data arrives at slow pace and there is
no much benefit from a better implementation.

Default choice is the implementation that uses two CUDA streams and overlaps
copy/compute phases. This implementation provides a better throughput when input
requests arrive at high frequency.

##### DLBS_TENSORRT_NO_POSIX_FADV_DONTNEED
Do not advise OS to not to cache dataset files. By default, data readers advise
OS that files they open will not be needed in future meaning that OS should not
cache these files. This can be used to emulate presence of a large dataset and
to benchmark storage. Export the following variable to disable this functionality:
```bash
export DLBS_TENSORRT_NO_POSIX_FADV_DONTNEED=1
```

##### DLBS_TENSORRT_DATASET_SPLIT
This environment variable defines how multiple readers split dataset files. By
default, this split is uniform (`uniform`). Each reader gets its own unique
collection of files. If there are more readers than files, some readers will not
get their files and benchmark application will exit.

Another option is when each reader reads entire dataset (`nosplit`) randomly
shuffling files during the startup:
```bash
export DLBS_TENSORRT_DATASET_SPLIT=nosplit
```
