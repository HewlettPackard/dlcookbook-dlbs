FROM ubuntu:16.04
MAINTAINER sergey.serebryakov@hpe.com

ARG build_space=/tmp

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python-pip \
        python-dev \
        git \
        cmake \
        libnuma-dev \
        wget && \
    wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | apt-key add - && \
    echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | tee /etc/apt/sources.list.d/rocm.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        rocm-dev \
        rocrand \
        rocblas \
        rocfft \
        miopengemm \
       && \
    rm -rf /var/lib/apt/lists/*

ENV ROC_SPARSE_URL https://github.com/ROCmSoftwarePlatform/rocSPARSE/releases/download/v0.1.4.1/rocsparse-0.1.4.1-Linux.deb
ENV HIP_SPARSE_URL https://github.com/ROCmSoftwarePlatform/hipSPARSE/releases/download/v0.1.4.0/hipsparse-0.1.4.0-Linux.deb
ENV MIOPEN_URL https://github.com/ROCmSoftwarePlatform/MIOpen/releases/download/1.7.0/MIOpen-HIP-1.7.0-49c48917-Linux.deb
RUN cd ${build_space} && \
    wget ${ROC_SPARSE_URL} && apt-get install ./rocsparse-* && rm ./rocsparse-* && \
    wget ${HIP_SPARSE_URL} && apt-get install ./hipsparse-*  && rm ./hipsparse-* && \
    wget ${MIOPEN_URL} && apt-get install ./MIOpen-* && rm ./MIOpen-*

RUN pip install --no-cache-dir setuptools wheel && \
    pip install --no-cache-dir enum pyyaml typing pybind11 numpy Pillow lmdb protobuf

ENV THRUST_HASHTAG e0b8fe2af3d345fb85689011140a20ff46fb610d
ENV HIP_PLATFORM hcc
RUN cd ${build_space} && git clone https://github.com/ROCmSoftwarePlatform/Thrust.git ./thrust && \
    cd ./thrust && git reset --hard ${THRUST_HASHTAG} && \
    git submodule update --init --recursive && \
    mkdir ./build && cd ./build && cmake .. && make install && \
    ln -s /usr/local/include/thrust/system/cuda/detail/cub /usr/local/include/cub && \
    cd ${build_space} && rm -rf ./thrust

ARG version=b710aee8c2ac2daa36e5143b00982b06746a4bf7
ENV hip_DIR /opt/rocm/hip/lib/cmake/hip
ENV hcc_DIR /opt/rocm/hcc/lib/cmake/hcc
RUN cd ${build_space} && git clone https://github.com/ROCmSoftwarePlatform/pytorch.git ./pytorch && \
    cd ./pytorch && \
    git reset --hard ${version} && git submodule update --init --recursive && \
    for req in $(cat requirements.txt); do pip install $req; done && \
    python tools/amd_build/build_pytorch_amd.py && \
    python tools/amd_build/build_caffe2_amd.py && \
    USE_ROCM=1 BUILD_TESTS=OFF python setup.py install && \
    cd ${build_space} && rm -rf ./pytorch

ENV APEX_HASHTAG 53e1b61a1e2498e66e4af9ff19e0bc55955b24b0
RUN cd ${build_space} && git clone https://github.com/NVIDIA/apex ./nvidia_apex && \
    cd ./nvidia_apex && \
    git reset --hard ${APEX_HASHTAG} && \
    python setup.py install && \
    cd ${build_space} && rm -rf ./nvidia_apex

RUN cd ${build_space} && git clone https://github.com/pytorch/vision.git ./pytorchvision && \
    cd ./pytorchvision && python setup.py install && \
    cd ${build_space} && rm -rf ./pytorchvision

