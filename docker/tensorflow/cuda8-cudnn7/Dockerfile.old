FROM gcr.io/tensorflow/tensorflow:latest-devel-gpu

MAINTAINER Sergey Serebryakov  <sergey.serebryakov@hpe.com>

WORKDIR /tensorflow

ENV CI_BUILD_PYTHON python
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV TF_NEED_CUDA 1
ENV TF_CUDA_COMPUTE_CAPABILITIES=3.0,3.5,5.2,6.0,6.1
ENV HTTP_PROXY http://web-proxy-pa.labs.hpecorp.net:8088
RUN sed -i 's/zlib-1.2.8/zlib-1.2.11/g' tensorflow/workspace.bzl && \
    sed -i 's/36658cb768a54c1d4dec43c3116c27ed893e88b02ecfcb44f2166f9c0b7f2a0d/c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1/g' tensorflow/workspace.bzl && \
    tensorflow/tools/ci_build/builds/configured GPU bazel build -c opt --config=cuda tensorflow/tools/benchmark:benchmark_model && \
    mkdir -p /usr/local/tensorflow/tools/benchmark/ && \
    cp ./bazel-bin/tensorflow/tools/benchmark/benchmark_model /usr/local/tensorflow/tools/benchmark/ && \
    rm -rf /root/.cache

WORKDIR /root

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

RUN ["/bin/bash"]
