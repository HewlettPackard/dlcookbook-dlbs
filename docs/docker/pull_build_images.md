# __Build/pull Docker Images__

## NVIDIA GPU Cloud
A good source of optimized docker images is the [NVIDIA GPU Cloud](https://ngc.nvidia.com). You need to signup for the service in order to pull images. NVIDIA provides images with latest versions of various frameworks. I have tested images 17.11.

## Using Dockerfiles from frameworks distributions
Majority of frameworks provide Dockerfiles.

## Building images with DLBS
We provide a number of Dockerfiles for various frameworks/software stacks. We also provide a bash script that automates building process. The folder [docker](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker) contains all dockerfiles and [build](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/docker/build.sh) script.

The structure of that folder is the following:
1. The folder itself contains subfolders. One subfolder for one framework. The name of a subfolder becomes an image name.
2. Every subfolder contains subfolders that contain Dockerfiles for various hardware/software stack. The names of those subfolders become image tag.
3. A prefix is added to an image name. Default value is `hpe`. User can specify their own prefix with `--prefix my_prefix` command line argument.

The name of an image thus becomes `$prefix/$framework:$tag`. The build script requires one mandatory parameter - path to a folder with Dockerfile to build. For instance, the following command:
```bash
./build.sh tensorflow/cuda8-cudnn6
```
builds a specific version of TensorFlow docker images (versions are defined in [versions](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/docker/versions) file) with Ubuntu 16.04, CUDA 8 and cuDNN 6. The image name is `hpe/tensorflow:cuda8-cudnn6`.

This is a list of docker files we provide (they are not optimized yet, we are working on it):
* benchmarks
  * [ethernet](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/benchmarks/ethernet)
  * [infiniband](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/benchmarks/infiniband)
* BVLC Caffe
  * [cuda8-cudnn6](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/bvlc_caffe/cuda8-cudnn6)
  * [cuda8-cudnn7](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/bvlc_caffe/cuda8-cudnn7)
  * [cuda9-cudnn7](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/bvlc_caffe/cuda9-cudnn7)
* Caffe2
  * [cuda8-cudnn6](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/caffe2/cuda8-cudnn6)
  * [cuda8-cudnn7](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/caffe2/cuda8-cudnn7)
  * [cuda9-cudnn7](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/caffe2/cuda9-cudnn7)
* Intel Caffe
  * [cpu](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/intel_caffe/cpu)
* MXNet
  * [cuda8-cudnn6](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/mxnet/cuda8-cudnn6)
  * [cuda8-cudnn7](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/mxnet/cuda8-cudnn7)
  * [cuda9-cudnn7](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/mxnet/cuda9-cudnn7)
* NVIDIA Caffe
  * [cuda8-cudnn6](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/nvidia_caffe/cuda8-cudnn6)
  * [cuda8-cudnn7](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/nvidia_caffe/cuda8-cudnn7)
  * [cuda9-cudnn7](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/nvidia_caffe/cuda9-cudnn7)
* TensorFlow
  * [cuda8-cudnn6](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/tensorflow/cuda8-cudnn6)
  * [cuda8-cudnn7](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/tensorflow/cuda8-cudnn7)
  * [cuda9-cudnn7](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/tensorflow/cuda9-cudnn7)
* TensorRT
  * [cuda8-cudnn6](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/tensorrt/cuda8-cudnn6)
* PyTorch
  * [cuda9-cudnn7](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/pytorch/cuda9-cudnn7)
