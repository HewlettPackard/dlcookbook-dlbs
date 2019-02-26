# TensorRT Inference Benchmark Tool
The TensorRT inference benchmark tool, or just _the tool_, is a part of [HPE Deep Learning Benchmarking Suite]
(https://hpe.com/software/dl-cookbook) intended to run inference benchmarks with TensorRT inference engine.
In particular, it can run with or without DLBS and can be used for:

1. Benchmarking a particular model on your hardware.
2. Identifying bottlenecks (CPU/GPU, PCIe, storage/network).
3. This tool also supports users models in Caffe (.prototxt) or ONNX (.onnx) formats.


### Installation
The latest docker images (starting 18.12)  are based on TensorRT images from NGC. Previous images (<= 18.11) are based on 
`nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04` that require users to manually download TensorRT .deb package. The `versions`
file in docker directory lists TensorRT packages required to build previous docker images. The deb packages need to be
copied in the corresponding folders containing `Dockerfile` files.


#### Docker containers
For  versions <= 18.11, copy TensorRT deb package to `${DLBS_ROOT}/docker/tensorrt/${VERSION}`. Then, go to `${DLBS_ROOT}/docker`
and run:
```bash
./build.sh tensorrt/${VERSION}
```
to build docker container (`dlbs/tensorrt:${VERSION}`). The `${VERSION}` here is the version of the container that
needs to be built. For instance, it can be `18.10`. In general, it is recommended to build latest version that includes
all optimizations. 

Type `./build.sh --help` to learn about additional optional input parameters.

The build script will build programs and will install them and will make them available on the PATH environment variable
inside container.

To build latest TensorRT docker images (>= 18.12), just go to `${DLBS_ROOT}/docker` abd run `./build.sh tensorrt/${VERSION}`.
There is no need to download TensorRT deb packages.

#### Bare metal
To build this project by yourself, you will need cmake, CUDA, boost program options, opencv 2 and TensorRT installed
in your system. Go to `${DLBS_ROOT}/src/tensorrt` and run the following commands:
```bash
mkdir ./build && cd ./build
cmake .. && make -j$(nproc)
```

#### Configuring build process
Read this [document](build.md) that describes build parameters affecting the benchmarking tool.

### Running benchmarks
The inference benchmark tool is tightly integrated with Deep Learning Benchmarking Suite and it's the most easiest way to
run benchmarks (framework is `tensorrt`). If you need to run benchmarks with real data, make sure you read the following:

1. Read about [dataset options](datasets.md) for this tool.
2. Convert images to a binary format with [images2tensors](images2tensors.md) tool.
3. Find out at what rate you can stream images from your storage with [benchmark_tensor_dataset](dataset_benchmarks.md) tool.
4. Find out at what rate you can stream images from host to device memory with [benchmark_host2device](benchmark_host2device.md) tool.
5. Finally, run benchmarks. Most of the time, so called [single-process multi-GPU inference benchmarks](sprocess_benchmarks.md)
   can provide best performance. In some situations involving large number of GPUs, multi-socker servers and high throughput models
   it may be beneficial to run two or four benchmark process pinning them to specific cores and taking into account CPU-GPU
   connectivity. We call this type of benchmarks [multi-process inference benchmarks](mprocess_benchmarks.md).




