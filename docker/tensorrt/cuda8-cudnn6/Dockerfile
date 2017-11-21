FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
# FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER sergey.serebryakov@hpe.com


COPY nv-tensorrt-repo-ubuntu1604-7-ea-cuda8.0_2.0.2-1_amd64.deb /


RUN dpkg -i /nv-tensorrt-repo-ubuntu1604-7-ea-cuda8.0_2.0.2-1_amd64.deb && \
    rm /nv-tensorrt-repo-ubuntu1604-7-ea-cuda8.0_2.0.2-1_amd64.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
        numactl \
        build-essential \
        cmake \
        git \
        wget \
        tensorrt-2 \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV TENSORRT_PRJ tensorrt2

RUN mkdir /${TENSORRT_PRJ} && \
    cd /${TENSORRT_PRJ} && \
    git init && \
    git remote add -f origin https://github.hpe.com/labs/dlcookbook.git && \
    git config core.sparseCheckout true && \
    echo "src/${TENSORRT_PRJ}" >> .git/info/sparse-checkout && \
    git pull origin master && \
    cd ./src/${TENSORRT_PRJ}/ && \
    mkdir build && \
    cd ./build/ && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/tensorrt .. && \
    make && \
    make install && \
    cd / && \
    rm -rf /${TENSORRT_PRJ}


ENV PATH /opt/tensorrt/bin:$PATH
