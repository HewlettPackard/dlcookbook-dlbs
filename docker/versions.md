# __Frameworks Versions__

This document outlines versions of frameworks that are now supported by the benchmarking tool by default. Versions are defined in versions [file](versions).

When building a docker image, the build [script](build.sh) will search for a particular version in that file. First, the script will search for a key `name/tag` in that file. For instance, if a user wants to build a TensorFlow image `hpe/tensorflow:cuda9-cudnn7`, the script will use `tensorflow/cuda9-cudnn7` as a key trying to get commit for a specific framework and tag. If there's no such key, the script will secondly try to search only for a framework name (`tensorflow` in the above example). If it fails to find it, it will not pass `--build-arg version=${version}` argument to a docker build command assuming the docker file defines a default version (usually, it's `master`). In other cases, if a version has been found, the build script will use it.

Users may provide their specific commit via a command line argument `--version` to a build script (build.sh) i.e. for instance `./build.sh --version A_SPECIFIC_COMMIT bvlc_caffe/cuda8-cudnn7`. In this case, this version will be used for ALL docker files specified by a user. This basically means a user-provided version should only be used when building one docker image.

## Brief summary

| Framework | Commit | Date | Version |
|-----------|--------|------|---------|
| Intel Caffe | [f6a2a6b](https://github.com/intel/caffe/commit/f6a2a6b05defab4b637028ce4f7719cac340a86d) | Dec 13, 2017 | 1.0.0-rc3 |
| BVLC Caffe | [c430690](https://github.com/BVLC/caffe/commit/c430690aa5528e94e019971b94de325539984e77) | Nov 7, 2017 | 1.0.0 |
| NVIDIA Caffe | [cdb3d9a](https://github.com/NVIDIA/caffe/commit/cdb3d9a5d46774a3be3cc4c4ecc0bcd760901cc1) | Sep 7, 2017 | 0.16.4 |
| MXNET | [079fc45](https://github.com/apache/incubator-mxnet/commit/079fc45383c2dad588ad6a1d779c33b738d514f8) | Nov 15, 2017 | 1.0.0 |
| Caffe2 | [f075a2b](https://github.com/caffe2/caffe2/commit/f075a2b55f93c6c9c7e5cca0c4279406a03b8653) | Oct 10, 2017 | 0.8.1 |
| TensorFlow (only CUDA9/cuDNN7)| [c877a42](https://github.com/tensorflow/tensorflow/commit/8d327187577c797499d5697cdef79af6a5fc7823) | March 1, 2018 | 1.6.0 |
| TensorRT | - | - | 2.0.0 |

#### Intel Caffe
- [f6a2a6b05defab4b637028ce4f7719cac340a86d](https://github.com/intel/caffe/commit/f6a2a6b05defab4b637028ce4f7719cac340a86d)
- [Project](https://github.com/intel/caffe/tree/f6a2a6b05defab4b637028ce4f7719cac340a86d)
- Dec 13, 2017

#### BVLC Caffe
- [c430690aa5528e94e019971b94de325539984e77](https://github.com/BVLC/caffe/commit/c430690aa5528e94e019971b94de325539984e77)
- [Project](https://github.com/BVLC/caffe/tree/c430690aa5528e94e019971b94de325539984e77)
- Nov 7, 2017
- Architectures are defined in docker files. It's Volta ready with CUDA 9.

#### NVIDIA Caffe
- [cdb3d9a5d46774a3be3cc4c4ecc0bcd760901cc1](https://github.com/NVIDIA/caffe/commit/cdb3d9a5d46774a3be3cc4c4ecc0bcd760901cc1)
- [Project](https://github.com/NVIDIA/caffe/tree/cdb3d9a5d46774a3be3cc4c4ecc0bcd760901cc1)
- Sep 7, 2017
- Architectures are defined in docker files. It's Volta ready with CUDA 9.

#### MXNET
- [079fc45383c2dad588ad6a1d779c33b738d514f8](https://github.com/apache/incubator-mxnet/commit/079fc45383c2dad588ad6a1d779c33b738d514f8)
- [Project](https://github.com/apache/incubator-mxnet/tree/079fc45383c2dad588ad6a1d779c33b738d514f8)
- Nov 15, 2017
- CUDA ARCH selection is [here](https://github.com/apache/incubator-mxnet/blob/079fc45383c2dad588ad6a1d779c33b738d514f8/Makefile#L250). Depending on CUDA version, supported architectures will be selected from this list - 30 35 50 52 60 61 70. It's Volta ready with CUDA 9.

#### Caffe2
- [f075a2b55f93c6c9c7e5cca0c4279406a03b8653](https://github.com/caffe2/caffe2/commit/f075a2b55f93c6c9c7e5cca0c4279406a03b8653)
- [Project](https://github.com/caffe2/caffe2/tree/f075a2b55f93c6c9c7e5cca0c4279406a03b8653)
- Oct 10, 2017
- CUDA ARCH selection is [here](https://github.com/caffe2/caffe2/blob/f075a2b55f93c6c9c7e5cca0c4279406a03b8653/cmake/Cuda.cmake). Depending on CUDA version, appropriate architectures will be selected, It's Volta ready with CUDA 9.

#### TensorFlow
- cuda9-cudnn7: [8d327187577c797499d5697cdef79af6a5fc7823](https://github.com/tensorflow/tensorflow/commit/8d327187577c797499d5697cdef79af6a5fc7823)
- other containers: [d752244fbaad5e4268243355046d30990f59418f](https://github.com/tensorflow/tensorflow/commit/d752244fbaad5e4268243355046d30990f59418f)
- [Project](https://github.com/tensorflow/tensorflow/tree/d752244fbaad5e4268243355046d30990f59418f)
- March 1, 2018
- Architectures are defined in docker files. It's Volta ready with CUDA 9.

> Currently, only `cuda9-cudnn7` container with github hash `8d327187577c797499d5697cdef79af6a5fc7823` cam work with latest tf_cnn benchmarks. The cuda8-cudnn7 and cuda8-cudnn6 containers can be built but will not work with TensorFlow benchmark backed.

#### TesnsorRT
- nv-tensorrt-repo-ubuntu1604-7-ea-cuda8.0_2.0.2-1_amd64.deb
- TensorRT version 2.0.0
- CUDA 8, cuDNN 5. It's not Volta ready.
