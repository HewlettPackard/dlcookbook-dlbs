# [MLPerf MLBox](https://github.com/mlperf/mlbox)
This document briefly presents a Deep Learning Benchmarking Suite and introduces new functionality for packaging benchmarks into self-contained machine learning boxes. 


## [Deep Learning Cookbook](https://developer.hpe.com/platform/hpe-deep-learning-cookbook/home)
- [HPE Deep Learning Benchmarking Suite](https://github.com/HewlettPackard/dlcookbook-dlbs) (DLBS). Automated benchmarking tool to collect performance of different deep learning workloads on various hardware and software configurations
- [HPE Deep Learning Performance Guide](http://dlpg.labs.hpe.com) (DLPG). A web-based tool to guide a choice of optimal hardware and software configuration via analysis of collected performance data and applying performance models.


## Deep Learning Benchmarking Suite
- DL/ML oriented benchmarking suite for hardware and software infrastructures
    - Frameworks: TensorFlow, Caffe, Caffe2, MXNet, PyTorch, TensorRT
    - Neural networks: 25 models (AlexNet, GoogleNet, ResNets, VGGs, ...)
    - User models via ONNX and framework-specific formats
    - Runtimes: bare metal / docker / singularity
    - Phases: training and inference.
    - Containers: reference docker files / NVIDIA GPU Cloud (NGC)
    - Resource monitor: GPU/CPU/memory utilization
    - API: command line interface / Python API
    - Supporting tools: host<->device bandwidth measurement / datasets / ...
- Target metrics
    - Iteration time / throughput (instances / second)


## Introduction
DLBS takes as input a benchmark configuration and outputs log files, one log file per benchmark. Example scripts that implement functionality described in this chapter are located [here](./01_introduction)

Install benchmarking suite:
```bash
git clone https://github.com/HewlettPackard/dlcookbook-dlbs.git ./dlbs
```

Initialize host environment and pull docker image:
```bash
cd ./dlbs  &&  source ./scripts/environment.sh
docker pull nvcr.io/nvidia/tensorflow:18.07-py3
```

Run 8 TensorFlow benchmarks (2 models and 4 batch sizes):
```
python $experimenter run -Pexp.framework='"nvtfcnn"' \
                         -Vexp.model='["resnet50", "alexnet_owt"]' \
                         -Vexp.replica_batch='[4, 8, 16, 32]' \
                         -Pexp.log_file='"${HOME}/dlbs/logs/${exp.model}_${exp.replica_batch}.log"' \
                         -Pexp.docker_image='"nvcr.io/nvidia/tensorflow:18.07-py3"'
```
All other parameters (benchmark phase, number of iterations, precision, ...) have their default values that can also be changed - all benchmark parameters will be stored in an output log file. Expected output:
```
Duration (minutes): ........... 4.7917343
  Total benchmarks in plan: ..... 8
  |--Inactive benchmarks: ....... 0
  |--Existing benchmarks: ....... 0
  |  |--Successful benchmarks: .. 0
  |  |--Failed benchmarks: ...... 0
  |--Active benchmarks: ......... 8
  |  |--Completed benchmarks: ... 8
  |  |--Successful benchmarks: .. 8
  |  |--Failed benchmarks: ...... 0

```

Parse summary and build benchmark report:
```bash
python $benchdata report ./logs --report regular
```
Expected output:
```
Batch time (milliseconds)
                            Replica Batch
  Model       DeviceType         4       8      16       32  
  AlexNetOWT  gpu            20.07   20.89   22.45    26.55  
  ResNet50    gpu            48.59   66.48   95.87   158.48  



Throughput (instances per second e.g. images/sec)
                            Replica Batch
  Model       DeviceType         4       8      16       32  
  AlexNetOWT  gpu           199.32  383.01  712.54  1205.46  
  ResNet50    gpu            82.33  120.34  166.90   201.91  



This report is configured with the following parameters:
 inputs = ['exp.model_title', 'exp.device_type']
 output = exp.replica_batch
 output_cols = None
 report_speedup = False
 report_efficiency = False

```
Full output is stored in this [file](./01_introduction/run.log)

## External benchmark controller
Put together benchmark configuration, let DLBS run it.
```bash
python $experimenter --config ./config.json
```  
JSON file is an alternative approach to provide benchmark parameters. Example JSON file is located [here](./02_json_configuration/config.json).
#### config.json
  - __benchmark_parameters__: `batch size`, `model name`, `input dataset`, `output files`, ...
  - __benchmark_backend__: `nvidia:nvtfcnn`, `dlbs:tensorrt`, `intel:openvino`, `google:tf_cnn_benchmarks`, `dlbs:mxnet`, ...
  - __benchmark_runtime__: `docker:container`, `singularity:container`, `host:python`

#### Pros
- Only benchmark configuration is required which is a small JSON file.
- Works OK with SLURM, PBSPro and other similar tools.
- Easy to modify benchmark configuration.

#### Cons
- User is expected to have basic knowledge about DLBS.
- User is responsible for using the right benchmark version (git commit tag).
- Does not work well with docker orchestration frameworks such as K8S.
- Multi-node benchmark protocol could be better. 

## Internal benchmark controller
DLBS can use local runtime (no docker), so some sort of benchmark packaging is required. Take benchmark configuration and package it into a self-contained docker image. It's a three step process:
- __Package benchmark__. `Input`: benchmark configuration (JSON file) + other parameters, `output` - docker file, helper build and run scripts.
- __Build docker image__. Use build script to build a docker image.
- __Publish docker image__. Push docker image to your local docker hub.

__Why docker file?__
- Send small docker files instead of sending large containers.
- Docker files can be stored on GitHub.
- Bonus feature - straightforward way to replace runtime assuming the runtime is provided by a base image. 

Naive example is located in this [folder](./03_mlbox):

Initialize host environment:
```bash
export ROOT_DIR=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
. ${ROOT_DIR}/../../../../scripts/environment.sh
```

Create MLBox:
```bash
builder=${DLBS_ROOT}/python/dlbs/mlbox/mlbox_builder.py
python ${builder} --config "./config.json"\
                  --hashtag "7c5ca5a6dfa4e2f7b8b4d81c60bd8be343dabd30"\
                  --work_dir "./mlbox"\
                  --base_image "nvcr.io/nvidia/tensorflow:18.07-py3"\
                  --docker_image "dlbs/ngc-tf:18.07-mlbox"\
                  --docker_launcher "nvidia-docker"
```

Build MLBox:
```bash
cd ./mlbox
./build.sh
```
Run it:
```bash
./run.sh
```
