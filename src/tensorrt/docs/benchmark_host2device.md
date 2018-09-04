# Benchmarks host-to-device data transfers

### Overview
Another possible bottleneck is the host to device data transfers usually done via
PCIe lanes. The `benchmark_host2device` tool copies data from host to GPU memory 
measuring achievable bandwidth.

### Command line arguments
Example usage of this tool looks liek this:
```bash
benchmark_host2device_copy --pinned \
                           --gpu 0 \
                           --num_warmup_batches 50 \
                           --num_batches 100 \
                           --size 128
```

where:
1. `--pinned`                 Allocate host pinned memory (else memory is pageable).
2. `--gpu ID`                 GPU identifier to use.
3. `--num_warmup_batches N`   Number of warm-up copy transfers.
4. `--num_batches M`          Number of benchamrk copy transfers.
5. `--size SIZE`              Size of a data in megabytes.

### Running benchmarks with DLBS
DLBS provides example script `tutorials/dlcookbook/tensorrt/benchmark_host2device.sh`
that helps with running dataset benchmarks with containers:
```bash
source ./scripts/environment.sh
script=./tutorials/dlcookbook/tensorrt/benchmark_host2device.sh
$script --gpu 0 --size 64 --pinned_mem true --num_warmup_batches 20 --num_batches 500 \
        --cpus 0-17 --docker_img dlbs/tensorrt:18.10
```
