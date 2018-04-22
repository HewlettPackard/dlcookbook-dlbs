FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
LABEL maintainer caffe-maint@googlegroups.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        numactl \
        build-essential \
        cmake \
        git \
        wget \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-pip && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/xianyi/OpenBLAS.git && cd ./OpenBLAS && \
    make -j"$(nproc)" && make install PREFIX=/opt/OpenBLAS && \
    cd .. && rm -rf ./OpenBLAS

# --upgrade pip
RUN pip install --no-cache-dir --upgrade setuptools wheel && \
    pip install --no-cache-dir \
        numpy \
        scipy


# This is where Caffe will be installed
ENV CAFFE_ROOT=/opt/caffe

# BVLC Caffe's version to clone
ARG version=master
ARG cuda_arch_bin="30 35 50 60 61 70"
ARG cuda_arch_ptx="50"
RUN git clone https://github.com/BVLC/caffe.git bvlc_caffe && cd ./bvlc_caffe && \
    git reset --hard ${version} &&\
    cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd .. && \
    git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j install && cd .. && rm -rf nccl && \
    mkdir build && cd build && \
    cmake -DCUDA_ARCH_NAME="Manual"\
          -DCUDA_ARCH_BIN="${cuda_arch_bin}"\
          -DCUDA_ARCH_PTX="${cuda_arch_ptx}"\
          -DUSE_CUDNN=1 \
          -DUSE_NCCL=1 \
          -DBLAS=open\
          -DBLAS_INCLUDE=/opt/OpenBLAS/include\
          -DBLAS_LIB=/opt/OpenBLAS/lib\
          -DCMAKE_INSTALL_PREFIX=$CAFFE_ROOT .. && \
    make -j"$(nproc)" && make install && \
    cd ../../ && rm -rf ./bvlc_caffe

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/bin:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/lib" >> /etc/ld.so.conf.d/caffe.conf && \
    echo "/opt/OpenBLAS/lib" >> /etc/ld.so.conf.d/caffe.conf && \
    ldconfig

WORKDIR /workspace
