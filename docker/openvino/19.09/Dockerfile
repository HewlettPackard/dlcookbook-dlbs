FROM ubuntu:18.04

# System dependencies
RUN apt-get update && apt-get -y upgrade && apt-get autoremove -y
RUN apt install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y --no-install-recommends \
        numactl \
        build-essential \
        cpio \
        curl \
        git \
        lsb-release \
        pciutils \
        python3.5 \
        python3.5-dev \
        python3-pip \
        python3-setuptools \
        sudo \
        mc \
        vim

ADD l_openvino_toolkit* /openvino/
ARG OPENVINO_DIR=/opt/intel/openvino

# OpenVINO dependencies
RUN cd /openvino/ && \
    ./install_openvino_dependencies.sh

RUN pip3 install wheel
RUN pip3 install numpy pyyaml

# OpenVINO itself
RUN cd /openvino/ && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh --silent silent.cfg

# Model Optimizer
RUN cd ${OPENVINO_DIR}/deployment_tools/model_optimizer/install_prerequisites && \
    ./install_prerequisites.sh

# Model ZOO
RUN cd ${OPENVINO_DIR}/deployment_tools/open_model_zoo/tools/downloader && \
    pip3 install -r ./requirements.in

# Benchmark APP (SHELL thing: https://stackoverflow.com/a/25423366)
SHELL ["/bin/bash", "-c"]
RUN mkdir /opt/intel/openvino_benchmark_app && \
    cd /opt/intel/openvino_benchmark_app && \
    source ${OPENVINO_DIR}/bin/setupvars.sh && \
    cmake -DCMAKE_BUILD_TYPE=Release ${OPENVINO_DIR}/deployment_tools/inference_engine/samples && \
    make -j"$(nproc)" benchmark_app
SHELL ["/bin/sh", "-c"]

# Clean up
RUN apt autoremove -y && \
    rm -rf /openvino /var/lib/apt/lists/*

ENV OPENVINO_DIR=${OPENVINO_DIR}

# For interactive shells (docker run -i ...).
RUN echo "source ${OPENVINO_DIR}/bin/setupvars.sh" >> /root/.bashrc

CMD ["/bin/bash"]
