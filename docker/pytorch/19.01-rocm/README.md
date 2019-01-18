# PyTorch for ROCm platform

A PyTorch dockerfile for ROCm platform. As of now (1/17/2019), there is no multi-GPU support because
RCCL library is stil not ready for production. This GitHub thread provides more details:
    https://github.com/ROCmSoftwarePlatform/pytorch/issues/31


### Building docker image
Standard recommended way to build this container is to use docker build:
```bash
export http_proxy=<YOUR_HTTP_PROXY>
export https_proxy=<YOUR_HTTP_PROXY>

cd <DLBS_ROOT>/docker
./build.sh --prefix dlbs pytorch/19.01-rocm
```

If you happen to have too little free space on root partition (what I had at the moment of experiments
with PyTorch for ROCm), here is the sort of bash script that can be used to build the container manually:
  - Pull `ubuntu:16.04`.
    ```bash
    docker pull ubuntu:16.04
    ```
  - Run container (you may want to adjust docker args for your system).
    ```bash
    docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --privileged -v /sys/class:/sys/class -v /dev/shm:/dev/shm ubuntu:16.04
    ```
  - Execute the script below, do not exit running container. You may want to use different build space (I use /dev/shm).
    ```bash
    export http_proxy=<YOUR_HTTP_PROXY>
    export https_proxy=<YOUR_HTTP_PROXY>

    apt-get update
    apt-get install -y --no-install-recommends build-essential python-pip python-dev git cmake libnuma-dev wget

    wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | apt-key add - && echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | tee /etc/apt/sources.list.d/rocm.list
    apt-get update
    apt-get install -y --no-install-recommends rocm-dev rocrand rocblas rocfft miopengemm
    rm -rf /var/lib/apt/lists/*

    export ROC_SPARSE_URL=https://github.com/ROCmSoftwarePlatform/rocSPARSE/releases/download/v0.1.4.1/rocsparse-0.1.4.1-Linux.deb
    export HIP_SPARSE_URL=https://github.com/ROCmSoftwarePlatform/hipSPARSE/releases/download/v0.1.4.0/hipsparse-0.1.4.0-Linux.deb
    export MIOPEN_URL=https://github.com/ROCmSoftwarePlatform/MIOpen/releases/download/1.7.0/MIOpen-HIP-1.7.0-49c48917-Linux.deb
    cd /dev/shm
    wget ${ROC_SPARSE_URL} && apt-get install ./rocsparse-* && rm ./rocsparse-*
    wget ${HIP_SPARSE_URL} && apt-get install ./hipsparse-*  && rm ./hipsparse-*
    wget ${MIOPEN_URL} && apt-get install ./MIOpen-* && rm ./MIOpen-*

    pip install --no-cache-dir setuptools wheel && pip install --no-cache-dir enum pyyaml typing pybind11 numpy Pillow lmdb protobuf

    export THRUST_HASHTAG=e0b8fe2af3d345fb85689011140a20ff46fb610d
    export HIP_PLATFORM=hcc
    cd /dev/shm && git clone https://github.com/ROCmSoftwarePlatform/Thrust.git ./thrust && cd ./thrust && git reset --hard ${THRUST_HASHTAG} && git submodule update --init --recursive
    mkdir ./build && cd ./build && cmake .. && make install
    ln -s /usr/local/include/thrust/system/cuda/detail/cub /usr/local/include/cub
    cd /dev/shm && rm -rf ./thrust

    export version=b710aee8c2ac2daa36e5143b00982b06746a4bf7 hip_DIR=/opt/rocm/hip/lib/cmake/hip hcc_DIR=/opt/rocm/hcc/lib/cmake/hcc
    cd /dev/shm && git clone https://github.com/ROCmSoftwarePlatform/pytorch.git ./pytorch && cd ./pytorch && git reset --hard ${version} && git submodule update --init --recursive
    pip install --no-cache-dir -r requirements.txt
    python tools/amd_build/build_pytorch_amd.py && python tools/amd_build/build_caffe2_amd.py
    USE_ROCM=1 BUILD_TESTS=OFF python setup.py install
    cd /dev/shm && rm -rf ./pytorch

    export APEX_HASHTAG=53e1b61a1e2498e66e4af9ff19e0bc55955b24b0
    cd /dev/shm && git clone https://github.com/NVIDIA/apex ./nvidia_apex && cd ./nvidia_apex && git reset --hard ${APEX_HASHTAG}
    python setup.py install
    cd /dev/shm && rm -rf ./nvidia_apex

    cd /dev/shm && git clone https://github.com/pytorch/vision.git ./pytorchvision && cd ./pytorchvision
    python setup.py install
    cd /dev/shm && rm -rf ./pytorchvision

    apt-get clean && rm -rf /var/lib/apt/lists/* && rm -rf ~/.cache/pip
    unset http_proxy https_proxy ROC_SPARSE_URL HIP_SPARSE_URL MIOPEN_URL THRUST_HASHTAG version APEX_HASHTAG
    ```
  - Open another shell and commit changes.
    ```bash
    docker ps
    docker commit <CONTAINER-ID>  dlbs/pytorch:19.01-rocm
    ```

### Running Benchmarks

This is the example configuration file:
```json
{
  "parameters": {
    "exp.framework": "pytorch",
    "exp.docker_image": "dlbs/pytorch:19.01-rocm",

    "exp.num_warmup_batches": 50,
    "exp.num_batches": 200,
    "exp.log_file": "${BENCH_ROOT}/logs/amd_mi25/${exp.framework}/${exp.dtype}/${exp.num_gpus}_${exp.model}_${exp.effective_batch}.log",

    "exp.dtype": "float32",

    "exp.ignore_past_errors": true,

    "runtime.python": "python",

    "exp.docker_launcher": "sudo docker",
    "exp.docker_args": "--rm --pid=host --ipc=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --privileged  -v /sys/class:/sys/class -v ${HOME}/.cache/miopen:/root/.cache/miopen -v ${HOME}/.config/miopen:/root/.config/miopen",

    "exp.status": ""
  },
  "variables":{
    "exp.model": ["resnet50"],
    "exp.gpus": ["0"],
    "exp.replica_batch": [128],
    "exp.trial": [1]
  }
}
```

