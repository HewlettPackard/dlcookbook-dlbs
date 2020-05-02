FROM dlbs/openvino:19.09

# This docker file needs complete redesign. It was written ad-hoc.
# The '2019_R3_DLBS' branch in my repo contains patch to disable VNNI instructions for CPUs where they are available.
# Use: export MKLDNN_NO_VNNI=1

# System dependencies
RUN apt-get update && apt-get -y upgrade && apt-get autoremove -y

ENV OPENVINO_DIR=/opt/intel/openvino

RUN git config --global http.proxy ${http_proxy} && git config --global https.proxy ${https_proxy}

RUN mkdir -p /root/workspace && cd /root/workspace &&\
    git clone https://github.com/sergey-serebryakov/dldt.git && cd ./dldt &&\
    git fetch && git checkout 2019_R3_DLBS &&\
    cd ./inference-engine/ && git submodule init && git submodule update --recursive &&\
	./install_dependencies.sh

ENV MKLML_PACKAGE=mklml_lnx_2019.0.3.20190220
RUN cd /root/workspace/dldt/inference-engine/thirdparty/mkl-dnn &&  mkdir ./external && cd ./external && \
    wget "https://github.com/intel/mkl-dnn/releases/download/v0.18/${MKLML_PACKAGE}.tgz" && \
    tar -xzf "${MKLML_PACKAGE}.tgz"

RUN	cd /root/workspace/dldt/inference-engine/thirdparty/mkl-dnn && \
    mkdir -p /opt/mklml_lnx && cp -R ./external/mklml_lnx_*/* /opt/mklml_lnx

# TBB is important! OMP works, but benchmark app hangs forever once a benchmark is completed.
RUN	cd /root/workspace/dldt/inference-engine &&\
    mkdir build && cd build &&\
    /bin/bash -c "source ${OPENVINO_DIR}/bin/setupvars.sh &&\
	              cmake -DTHREADING=TBB -DMKLROOT=/opt/mklml_lnx -DGEMM=MKL -DENABLE_MKL_DNN=ON -DENABLE_CLDNN=OFF\
	                    -DENABLE_GNA=OFF -DENABLE_SAMPLES=OFF -DENABLE_VPU=OFF -DENABLE_MYRIAD=OFF\
	                    -DENABLE_SAMPLES_CORE=OFF -DCMAKE_BUILD_TYPE=Release -DENABLE_PLUGIN_RPATH=ON .. &&\
                  make -j$(nproc)"

RUN cd /root/workspace/dldt/inference-engine &&\
    cp ./bin/intel64/Release/lib/libMKLDNNPlugin.so ${OPENVINO_DIR}/deployment_tools/inference_engine/lib/intel64

CMD ["/bin/bash"]
