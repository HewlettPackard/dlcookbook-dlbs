FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER sergey.serebryakov@hpe.com

# https://github.com/ContinuumIO/libhdfs3-downstream/tree/master/libhdfs3

ARG version=nv-tensorrt-repo-ubuntu1604-cuda9.0-trt5.0.0.10-rc-20180906_1-1_amd64.deb
ENV TENSORRT_PACKAGE=$version

COPY $version /

RUN dpkg -i /${TENSORRT_PACKAGE} && \
    rm /${TENSORRT_PACKAGE} && \
    apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	cmake \
	git \
	wget \
	libboost-dev \
	libboost-program-options-dev \
	protobuf-compiler \
	libprotobuf-dev \
	libxml2-dev \
	libssl-dev \
	libkrb5-dev \
	libcurl4-openssl-dev \
	libgsasl7-dev \
	uuid-dev \
	numactl \
	doxygen \
	graphviz \
	tensorrt \
	python-libnvinfer-doc \
	uff-converter-tf \
	libopencv-dev \
	ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install GoogleTest and GoogleMock
RUN cd /tmp && git clone https://github.com/google/googletest.git && \
    cd ./googletest && mkdir build && cd ./build && cmake .. && \
    make -j$(nproc) && make install && \
    cd /tmp && rm -rf ./googletest

# Install LIBHDFS3
RUN cd /tmp && git clone https://github.com/ContinuumIO/libhdfs3-downstream.git libhdfs3 && \
    cd ./libhdfs3/libhdfs3 && mkdir build && cd build && ../bootstrap --prefix=/usr/local && \
    make -j$(nproc) && make install && \
    cd /tmp && rm -rf ./libhdfs3

COPY /tensorrt /tensorrt

RUN cd /tensorrt && rm -rf -- ./build && \
    mkdir build && cd ./build/ && \
    cmake -DHOST_DTYPE=INT8 -DCMAKE_INSTALL_PREFIX=/opt/tensorrt .. && \
    make -j$(nproc) && make build_docs && make install && \
    cd / && rm -rf /tensorrt


ENV PATH /opt/tensorrt/bin:$PATH
