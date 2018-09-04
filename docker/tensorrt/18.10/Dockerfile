FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER sergey.serebryakov@hpe.com

ARG version=nv-tensorrt-repo-ubuntu1604-ga-cuda9.0-trt3.0.4-20180208_1-1_amd64.deb
ENV TENSORRT_PACKAGE=$version

COPY $version /

# Should be a temporary solution. Something does not work when these repo are enabled (as of 3rd April 2018)
RUN cp /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/cuda.list.bak && \
    cp /etc/apt/sources.list.d/nvidia-ml.list /etc/apt/sources.list.d/nvidia-ml.list.bak && \
    printf '#%s' "$(cat /etc/apt/sources.list.d/cuda.list)" >/etc/apt/sources.list.d/cuda.list && \
    printf '#%s' "$(cat /etc/apt/sources.list.d/nvidia-ml.list)" >/etc/apt/sources.list.d/nvidia-ml.list
#    apt-get update && apt-get install -y --no-install-recommends apt-transport-https && \
#    mv /etc/apt/sources.list.d/cuda.list.bak /etc/apt/sources.list.d/cuda.list && \
#    mv /etc/apt/sources.list.d/nvidia-ml.list.bak /etc/apt/sources.list.d/nvidia-ml.list

RUN dpkg -i /${TENSORRT_PACKAGE} && \
    rm /${TENSORRT_PACKAGE} && \
    apt-get update && apt-get install -y --no-install-recommends \
        numactl \
        build-essential \
        cmake \
        git \
        wget \
        doxygen \
        graphviz \
        libprotobuf-dev \
        protobuf-compiler \
        tensorrt \
        python-libnvinfer-doc \
        uff-converter-tf \
        libboost-program-options-dev \
        libopencv-dev \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY /tensorrt /tensorrt

RUN cd /tensorrt && rm -rf -- ./build && \
    mkdir build && cd ./build/ && \
    cmake -DHOST_DTYPE=INT8 -DCMAKE_INSTALL_PREFIX=/opt/tensorrt .. && \
    make -j$(nproc) && make build_docs && make install && \
    cd / && rm -rf /tensorrt


ENV PATH /opt/tensorrt/bin:$PATH
