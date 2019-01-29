FROM nvcr.io/nvidia/tensorrt:18.11-py3
MAINTAINER sergey.serebryakov@hpe.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        numactl \
        build-essential \
        cmake \
        git \
        wget \
        doxygen \
        graphviz \
        libboost-program-options-dev \
        libopencv-dev \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY /tensorrt /tmp/tensorrt
# Why do not I remove /tmp/tensorrt folder?
# Explanation is here: https://github.com/moby/moby/issues/27214
RUN cd /tmp/tensorrt && rm -rf -- ./docker_build && \
    mkdir ./docker_build && cd ./docker_build && \
    cmake -DHOST_DTYPE=INT8 -DCMAKE_INSTALL_PREFIX=/opt/tensorrt .. && \
    make -j$(nproc) && make build_docs && make install && \
    cd /tmp/tensorrt && rm -rf ./docker_build


ENV PATH /opt/tensorrt/bin:$PATH
