FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
MAINTAINER sergey.serebryakov@hpe.com

# There's one issue with multithreaded OpenBLAS.

ENV NCCL_DEB=nccl-repo-ubuntu1604-2.0.5-ga-cuda8.0_2-1_amd64.deb
COPY $NCCL_DEB /

RUN dpkg -i /$NCCL_DEB && rm /$NCCL_DEB && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        automake \
        autoconf \
        libtool \
        nasm \
        libjemalloc-dev \
        git \
        liblapack-dev \
        libopenblas-dev \
        libopencv-dev \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        libnccl2 \
        libnccl-dev \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*


ENV JPEG_TURBO_ROOT=/opt/libjpeg-turbo
RUN cd /tmp && git clone https://github.com/libjpeg-turbo/libjpeg-turbo.git ./libjpeg-turbo && \
    cd ./libjpeg-turbo && autoreconf -fiv && \
    mkdir ./build && cd ./build && \
    sh ../configure CFLAGS="-O3 -fPIC" --prefix=$JPEG_TURBO_ROOT --libdir=$JPEG_TURBO_ROOT/lib && \
    make -j"$(nproc)" && make install && \
    echo "$JPEG_TURBO_ROOT/lib" >> /etc/ld.so.conf.d/libjpeg-turbo.conf && ldconfig && \
    cd /tmp && rm -rf ./libjpeg-turbo


#With NCCL1 mxnet hangs forever.
#ENV NCCL_ROOT=/opt/nccl
#RUN cd /tmp && git clone https://github.com/NVIDIA/nccl.git ./nccl && \
#    cd ./nccl && make CUDA_HOME=/usr/local/cuda && \
#    mkdir -p $NCCL_ROOT/include && mkdir -p $NCCL_ROOT/lib && \
#    cp ./build/include/nccl.h $NCCL_ROOT/include/ && \
#    cp ./build/lib/* $NCCL_ROOT/lib/ && \
#    echo "$NCCL_ROOT/lib" >> /etc/ld.so.conf.d/nccl.conf && ldconfig && \
#    cd .. && rm -rf ./nccl

ENV MXNET_ROOT=/opt/mxnet
ARG version=master
RUN git clone --recursive https://github.com/apache/incubator-mxnet $MXNET_ROOT && \
    cd $MXNET_ROOT && \
    git reset --hard ${version} && git submodule update && \
    make USE_CUDNN=1 USE_BLAS=openblas USE_CUDA=1 USE_NCCL=1 \
         USE_OPERATOR_TUNING=1 USE_LAPACK=1 USE_JEMALLOC=1 \
         USE_DIST_KVSTORE=0 USE_OPENMP=1 USE_OPENCV=1 USE_THREADED_ENGINE=1 \
         USE_CUDA_PATH=/usr/local/cuda \
         USE_LIBJPEG_TURBO=1 USE_LIBJPEG_TURBO_PATH=$JPEG_TURBO_ROOT -j$(nproc) && \
    cd ./python && python ./setup.py build && python ./setup.py install


WORKDIR /workspace
