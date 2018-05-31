# Inference Benchmarks
This tool was written to run inference benchmarks with NVIDIA's TensorRT. It can
server multiple purposes:

1. Benchmark a particular model on your hardware.
2. Identify bottlenecks (CPU/GPU, PCIe, storage/network).


### Installation
The inference benchmark tool can be built in docker container or bare metal. In both
cases, you need to download/install TensorRT package. We currently use version 3.0.4
and make sure you have the following packages downloaded/installed:
```
nv-tensorrt-repo-ubuntu1604-ga-cuda9.0-trt3.0.4-20180208_1-1_amd64.deb
```

#### Docker containers
Copy TensorRT deb package to `${DLBS_ROOT}/docker/tensorrt/cuda9-cudnn7`. Then, go
to `${DLBS_ROOT}/docker` and run
```bash
./build.sh tensorrt/cuda9-cudnn7
```
to build docker container (`hpe/tensorrt:cuda9-cudnn7`). Type `./build.sh --help`
to learn about additional optional input parameters.

The build script will build programs and will install them and will make them
available on the PATH environment variable inside container.

#### Bare metal
To build this project by yourself, you will need cmake, CUDA, boost program
options, opencv 2 and TensorRT 3.0.4 installed in your system. Go to
`${DLBS_ROOT}/src/tensorrt` and run the following commands:
```bash
mkdir ./build && cd ./build
cmake .. && make -j$(nproc)
```


### Running benchmarks
The inference benchmark tool is tightly integrated with Deep Learning Benchmarking
Suite and it's the most easiest way to run benchmarks (framework is `tensorrt`).
If you need to run benchmarks with real data, make sure you read the following:

1. Read about [dataset options](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/src/tensorrt/docs/datasets.md) for this tool
2. Convert images to a binary format with [images2tensors](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/src/tensorrt/docs/images2tensors.md)
   tool.
3. Find out at what rate you can stream images from your storage with
   [benchmark_tensor_dataset](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/src/tensorrt/docs/dataset_benchmarks.md) tool.
4. Additionally, the inference benchmark tool can be configured with
   [environment variables](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/src/tensorrt/docs/inference_benchmarks.md).

For overview of Deep Learning Benchmarking Suite, read this
[introduction](https://hewlettpackard.github.io/dlcookbook-dlbs/#/). We have a
number of example bash scripts that demonstrate how to run benchmarks. They
are located [here](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/tutorials/dlcookbook/tensorrt).
