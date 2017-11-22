# __TensorRT Benchmarking Tool__

This C++ project is based on NVIDIA's `giexec.cpp` sample shipped with
TensorRT distribution package.

It is used as a backend benchmarking tool for TensorRT inference framework.

## Compiling

To compile this project in a host OS, you need CUDA, cuDNN, Boost program options
and TensorRT package installed.
```bash
cd ./src/tensorrt
mkdir ./build && cd ./build
cmake -DCMAKE_INSTALL_PREFIX=/opt/tensorrt ..
make && make install

export PATH=/opt/tensorrt/bin:$PATH
```

## Docker container

In [docker](../../docker) folder the `build.sh` script can build Ubuntu-based
TensorRT containers. To build this, you need to have and copy the TensorRT
deb package to a docker root context folder.

In particular, now the [versions](../../docker/versions) file defines
one container `tensorrt/cuda8-cudnn6` with this tensorrt distribution `nv-tensorrt-repo-ubuntu1604-7-ea-cuda8.0_2.0.2-1_amd64.deb` (does it actually
require cudnn5?). So, to be able to build `tensorrt/cuda8-cudnn6` i.e. by executing
this command line:
```bash
./build.sh tensorrt/cuda8-cudnn6
```

you need to sign up and request access to TensorRT, then download this particular
version and copy this `deb` package to [tensorrt/cuda8-cudnn6](../../docker/tensorrt/cuda8-cudnn6)
context folder.

## Standalone run

The `tensorrt` tool accepts the following parameters:

1. `--version` Prints TensorRT version (i.e. 2.0.0)
2. `--model` Caffe's deployment (inference) prototxt descriptor
3. `--batch_size` Batch size to benchmark
4. `--dtype` Type of data - float32(float), float16 or int8
5. `--num_warmup_batches` Number of warmup iterations (is it required for TensorRT?)
6. `--num_batches` Number of benchmark iterations
7. `--profile` Print a layer-wise stats
8. `--input` Name of an input blob (default is `data`)
8. `--output` Name of an output blob (default is `prob`)


To select GPU, use CUDA_VISIBLE_DEVICES environmental variable i.e.:
```bash
CUDA_VISIBLE_DEVICES=1 tensorrt ...
```

## Benchmark results

**results.inference_time**

Average inference time excluding time to transfer data to/from GPU memory

**results.inference_times**

Inference times for every batch excluding time to transfer data to/from GPU memory

**results.total_time**

Average inference time including time to transfer data to/from GPU memory

**results.total_times**

Inference times for every batch including time to transfer data to/from GPU memory


These results are stored in standard key/value format and can be parsed with
log parser [tool](../../python/dlbs/logparser.py).
