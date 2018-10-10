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
        python3-dev \
        python3-pip \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    cd /usr/local/bin && \
    ln -s /usr/bin/python3 python


RUN pip3 install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade setuptools wheel && \
    pip install numpy Pillow lmdb protobuf

RUN git clone --recursive https://github.com/pytorch/pytorch /opt/pytorch && \
    cd /opt/pytorch && \
    git reset --hard ${version} && git submodule update && \
    for req in $(cat requirements.txt); do pip install $req; done && \
    NO_TEST=1 python setup.py install && \
    cd / && rm -rf /opt/pytorch

ENV APEX_HASHTAG 53e1b61a1e2498e66e4af9ff19e0bc55955b24b0
RUN git clone https://github.com/NVIDIA/apex /tmp/nvidia_apex && \
    cd /tmp/nvidia_apex && \
    git reset --hard ${APEX_HASHTAG} && \
    python setup.py install && \
    cd / && rm -rf /tmp/nvidia_apex

RUN git clone https://github.com/pytorch/vision.git /tmp/pytorchvision && \
    cd /tmp/pytorchvision && python setup.py install && \
    cd / && rm -rf /tmp/pytorchvision
