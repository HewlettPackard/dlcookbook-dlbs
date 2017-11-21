FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER sergey.serebryakov@hpe.com

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
	           numactl \
            libjemalloc-dev \
	           build-essential \
	           cmake \
            wget \
            git \
            libatlas-base-dev \
            libopencv-dev \
            python-dev \
            python3-dev \
	           ca-certificates && \
	   rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/xianyi/OpenBLAS.git && cd ./OpenBLAS && \
    make -j"$(nproc)" && make install PREFIX=/opt/OpenBLAS && \
    cd .. && rm -rf ./OpenBLAS

# the version of the pip shipped with ubuntu may be too lower, install a recent version here
RUN cd /tmp && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    python2 get-pip.py && \
    pip2 install nose pylint numpy nose-timer requests && \
    pip3 install nose pylint numpy nose-timer requests

ENV MXNET_ROOT=/opt/mxnet

ARG version=master
ARG cuda_arch_bin="30 35 50 60 61 70"
# See Makefile to enable CUDA_ARCH='$cuda_arch_bin'
ENV BUILD_OPTS USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 USE_DIST_KVSTORE=1 USE_JEMALLOC=1 USE_BLAS=openblas USE_OPENCV=1 ADD_CFLAGS="-I/opt/OpenBLAS/include" ADD_LDFLAGS="-L/opt/OpenBLAS/lib"
# ENV CLONE_TAG v0.10.0
RUN git clone --recursive https://github.com/apache/incubator-mxnet $MXNET_ROOT && cd $MXNET_ROOT && \
    git reset --hard ${version} && git submodule update && \
    make -j$(nproc) $BUILD_OPTS

ENV PYTHONPATH=$MXNET_ROOT/python:$PYTHONPATH
RUN echo "/opt/OpenBLAS/lib" >> /etc/ld.so.conf.d/mxnet.conf && \
    ldconfig

WORKDIR /workspace
