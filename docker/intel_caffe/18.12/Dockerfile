FROM ubuntu:16.04
MAINTAINER caffe-maint@googlegroups.com

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

RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade setuptools wheel  && \
    pip install --no-cache-dir \
        numpy \
	scipy

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# BVLC Caffe's version to clone
ARG version=master
RUN git clone https://github.com/intel/caffe.git . && \
    git reset --hard ${version} &&\
    for req in $(cat python/requirements.txt) pydot; do pip install $req; done && \
    mkdir build && cd build && \
    cmake -DCPU_ONLY=1 -DUSE_MKL2017_AS_DEFAULT_ENGINE=1 -DCMAKE_BUILD_TYPE=Release .. && \
    make all -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace
