FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
MAINTAINER sergey.serebryakov@hpe.com

ARG version=nv-tensorrt-repo-ubuntu1604-7-ea-cuda8.0_2.0.2-1_amd64.deb

COPY $version /

RUN dpkg -i /nv-tensorrt-repo-ubuntu1604-7-ea-cuda8.0_2.0.2-1_amd64.deb && \
    rm /nv-tensorrt-repo-ubuntu1604-7-ea-cuda8.0_2.0.2-1_amd64.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
        numactl \
        build-essential \
        cmake \
        git \
        wget \
        tensorrt-2 \
        libboost-program-options-dev\
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*


COPY /tensorrt /tensorrt

RUN cd /tensorrt && rm -rf -- ./build && \
    mkdir build && cd ./build/ && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/tensorrt .. && \
    make && make install && \
    cd / && rm -rf /tensorrt


ENV PATH /opt/tensorrt/bin:$PATH
