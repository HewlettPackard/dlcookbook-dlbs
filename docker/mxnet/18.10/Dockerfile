FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER sergey.serebryakov@hpe.com

# There's one issue with multithreaded OpenBLAS.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        build-essential \
        cmake \
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
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*


ENV JPEG_TURBO_ROOT=/opt/libjpeg-turbo
RUN cd /tmp && git clone https://github.com/libjpeg-turbo/libjpeg-turbo.git ./libjpeg-turbo && \
    cd ./libjpeg-turbo && mkdir ./build && cd ./build && \
    cmake -DENABLE_SHARED=true -DENABLE_STATIC=false -DCMAKE_C_FLAGS="-O3 -fPIC" -DCMAKE_INSTALL_PREFIX=$JPEG_TURBO_ROOT .. && \
    make -j"$(nproc)" && make install && ln -s $JPEG_TURBO_ROOT/lib64 $JPEG_TURBO_ROOT/lib && \
    echo "$JPEG_TURBO_ROOT/lib64" >> /etc/ld.so.conf.d/libjpeg-turbo.conf && \
    echo "$JPEG_TURBO_ROOT/lib" >> /etc/ld.so.conf.d/libjpeg-turbo.conf && \
    ldconfig && \
    cd /tmp && rm -rf ./libjpeg-turbo


ENV MXNET_ROOT=/opt/mxnet
ARG version=master
RUN git clone https://github.com/apache/incubator-mxnet $MXNET_ROOT && \
    cd $MXNET_ROOT && \
    git reset --hard ${version} && git submodule update --init --recursive && \
    make USE_CUDNN=1 USE_BLAS=openblas USE_CUDA=1 USE_NCCL=1 \
         USE_OPERATOR_TUNING=1 USE_LAPACK=1 USE_JEMALLOC=1 \
         USE_DIST_KVSTORE=1 USE_OPENMP=1 USE_OPENCV=1 USE_THREADED_ENGINE=1 \
         USE_CUDA_PATH=/usr/local/cuda \
         USE_LIBJPEG_TURBO=1 USE_LIBJPEG_TURBO_PATH=$JPEG_TURBO_ROOT -j$(nproc) && \
    cd ./python && python ./setup.py build && python ./setup.py install


WORKDIR /workspace
