FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
MAINTAINER sergey.serebryakov@hpe.com

# FIXME: Install OpenBLAS and compile numpy from sources linking with OpenBLAS.
# FIXME: What about OpenCV?
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
        python-numpy \
        python-pip \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/xianyi/OpenBLAS.git && cd ./OpenBLAS && \
    make -j"$(nproc)" && make install PREFIX=/opt/OpenBLAS && \
    cd .. && rm -rf ./OpenBLAS

# This is where Caffe will be installed
ENV CAFFE_ROOT=/opt/caffe

# NVIDIA Caffe's version to clone
ARG version=master
ARG cuda_arch_bin="30 35 50 60 61"
ARG cuda_arch_ptx="50"

# Sergey: I did not have time to figure this out, on some systems I need to specify paths to NVML library manually.
#         You will need this if you get bunch of errors saying nvml* functions are not resolved.
#cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 -DNDEBUG=1 -DNVML_FOUND=1 -DNVML_LIBRARY=/usr/local/cuda/lib64/stubs/libnvidia-ml.so -DNVML_INCLUDE_DIR=/usr/local/cuda/include .. && \
#cmake -DUSE_CUDNN=1 -DUSE_NCCL=1 -DNDEBUG=1 -DNVML_FOUND=0 .. && \
RUN git clone https://github.com/NVIDIA/caffe nvidia_caffe && cd ./nvidia_caffe && \
    git reset --hard ${version} &&\
    pip install --upgrade pip && \
    cd python && for req in $(cat requirements.txt) pydot; do pip install $req; done && pip install scikit-image && cd .. && \
    git clone https://github.com/NVIDIA/nccl.git && cd nccl && make -j install && cd .. && rm -rf nccl && \
    mkdir build && cd build && \
    cmake -DCUDA_ARCH_NAME="Manual"\
          -DCUDA_ARCH_BIN="${cuda_arch_bin}"\
          -DCUDA_ARCH_PTX="${cuda_arch_ptx}"\
          -DUSE_CUDNN=1\
          -DUSE_NCCL=1\
          -DNDEBUG=1\
          -DNVML_FOUND=1\
          -DNVML_LIBRARY=/usr/local/cuda/lib64/stubs/libnvidia-ml.so\
          -DNVML_INCLUDE_DIR=/usr/local/cuda/include\
          -DBLAS=open\
          -DBLAS_INCLUDE=/opt/OpenBLAS/include\
          -DBLAS_LIB=/opt/OpenBLAS/lib\
          -DCMAKE_INSTALL_PREFIX=$CAFFE_ROOT .. && \
    make -j"$(nproc)" && make install && \
    cd ../../ && rm -rf ./nvidia_caffe

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/bin:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/lib" >> /etc/ld.so.conf.d/caffe.conf && \
    echo "/opt/OpenBLAS/lib" >> /etc/ld.so.conf.d/caffe.conf && \
    ldconfig

WORKDIR /workspace
