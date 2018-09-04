# Dataset Benchmarks

### Overview
In order to warm-up system and/or benchmark storage/network, a tool named
`benchmark_tensor_dataset` can be used. This tool can be used to identify
maximal performance in terms of number of images/sec an inference benchmark
can stream from some storage to a host memory assuming inference itself is not
performed. In general, if an inference benchmark reports better throughput than
achieved with this test, this can be a signal that OS has cached some files and
cache is the place where benchmark was streaming data from.

This tools works with datasets built with [images2tensors](images2tensors.md) tool.

### Command line arguments
The following command line arguments are supported:
1. `--data_dir` Path to a dataset to use.
2. `--batch_size` Create batches of this size.
3. `--img_size` Size of images in a dataset (width = height).
4. `--num_prefetchers` Number of prefetchers (data readers).
5. `--prefetch_pool_size` Number of pre-allocated batches. Memory for batches is
   preallocated in advance and then reused by prefetchers.
6. `--num_warmup_batches` Number of warmup iterations.
7. `--num_batches` Number of benchmark iterations.
8. `--dtype` Tensor data type in the dataset- 'float' or 'uchar'.

For instance:
```bash
benchmark_tensor_dataset --data_dir=/mnt/data/imagenet100k/tensorrt --batch_size=512 \
                         --dtype=uchar --img_size=227 --num_prefetchers=3 \
                         --prefetch_pool_size=9 --num_warmup_batches=1000 \
                         --num_batches=5000
```

If a benchmark runs on a multi-socket machine and streams data from a network attached
storage, you may want to use `numactl` to pin benchmark process to the closest
CPU and also enforce local memory allocations, e.g.:
```bash
numactl --localalloc --physcpubind 0-17 benchmark_tensor_dataset ...
```

### Running benchmarks with DLBS
DLBS provides example script `tutorials/dlcookbook/tensorrt/benchmark_tensor_dataset.sh`
that helps with running dataset benchmarks with containers:
```bash
source ./scripts/environment.sh
script=./tutorials/dlcookbook/tensorrt/benchmark_tensor_dataset.sh
$script --data_dir /mnt/data/imagenet100k/tensorrt \
        --dtype uchar --img_size 227 \
        --batch_size 512 --num_prefetchers 8 \
        --num_preallocated_batches 32 \
        --num_warmup_batches 2000 --num_batches 8000
```