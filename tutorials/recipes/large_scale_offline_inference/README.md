# Large scale offline inference recipe
This recipe is intended to benchmark inference capabilities of a hardware including
ability of a storage to provide data at required throughput. This recipe benchmarks
throughput oriented workloads. Not all steps presented here are mandatory for getting
inference profile of your hardware.

All benchmarks are based on TensorRT benchmark backend described [here](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/src/tensorrt/docs/index.md).

Before running benchmarks, study `config.sh.example` and `config.json.example` files. You need to
create copies of these files (without `.example`) extension and adjust them accordingly.
The `config.json` file defines inference benchmark configurations and by default is suitable
for servers with 8 GPUs. Replica batch sizes defined there should be OK for 16 GB GPUs and above
(scripts were tested with V100 16 GB GPUs).


- `01` Prerequisites: build docker image (mandatory) and create dataset (optionally).
   - Build instructions are located [here](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/src/tensorrt/docs/index.md#installation). This recipe was tested with docker image `dlbs/tensorrt:18.10`.
   - If you need to run benchmarks with real dataset, adjust and run `./01/make_dataset`.
   - If you will need to run benchmarks with real data in memory and you have NUMA system,
     example script `./01/copy2memory` shows how to correctly copy data. Almost in all cases,
     you will need to adjust that file - read comments there.
- `02` Benchmark host-to-device throughput.
   - Measure host-to-device bandwidth. Will give intuition at what speed this software can
     stream data from host into GPU memory. Dataset is not required, GPU is required.
- `03` Run benchmarks with real/synthetic data. For each configuration of interest (synthetic data, real data in memory, real data on SSD etc.) I recommend creating copy of this folder e.g.:
  - `03a` Synthetic data (no real dataset).
  - `03b` Dataset is in /dev/shm.
  - `03c` Dataset is on local NVMe drive.
  - `03d` Dataset is on a network-attached storage.
  - `03e` Dataset is on some high performance storage.

  Every benchmark with real data generally includes three procedures:
  - Benchmark dataset readers (measure storage-to-host bandwidth). Results will tell you at what speed this software can stream data from storage into host memory. In most cases this should be very close to peak performance. This step does not need to be taken for synthetic data. Study `./03/a.storage2host_benchmark` file.
  - Benchmark inference/storage using single DLBS process. Will provide good performance with 1, 2 and probably 4 GPUs. With more GPUs in a NUMA system, multi-process benchmarks provide best performance. Study `./03/b.sprocess.run` file.
  - Benchmark inference/storage using multiple DLBS processes (two or four). This makes sense if you have >= 4 GPUs in a NUMA system (e.g. with 2 or more processes). For best results, you must know exact configuration of your hardware (number of CPUs/cores, hyper threading availability and how GPUs are connected to CPUs). Also, you will probably need to try several different configurations to find out the one that provides best performance. Study `./03/c.mprocess.run` file.
