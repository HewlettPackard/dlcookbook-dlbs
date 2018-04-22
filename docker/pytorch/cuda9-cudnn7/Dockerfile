FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER sergey.serebryakov@hpe.com

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        cmake \
        liblapack-dev \
        libopenblas-dev \
        libopencv-dev \
        python-dev \
        python-pip \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade setuptools wheel && \
    pip install numpy Pillow lmdb protobuf

ENV PYTORCH_ROOT=/opt/pytorch
RUN git clone --recursive https://github.com/pytorch/pytorch $PYTORCH_ROOT && \
    cd $PYTORCH_ROOT && \
    git reset --hard ${version} && git submodule update && \
    for req in $(cat requirements.txt); do pip install $req; done && \
    python setup.py install && \
    cd / && rm -rf ${PYTORCH_ROOT}



RUN git clone https://github.com/pytorch/vision.git /pytorchvision && \
    cd /pytorchvision && python setup.py install && \
    cd / && rm -rf /pytorchvision
