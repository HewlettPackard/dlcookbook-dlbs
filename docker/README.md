# Docker Images

We have choosen Docker containers as a primary mechanism to run benchmarks. There are several reasons to do that:

1. Minimal intervention and dependency requirements to the machinse where benchmarks run. There are many different hardware most of which we do not control, have regular access etc. Those machines are managed by other teams and we do not want to install/reinstall packages that are only required for benchmarks/experiments.
2. Benchmarks are reproducible. Setup environment once, run same code in same environment everywhere on variety of hardware.
3. Same performance as running in host environment.
4. Ease of use. Clone repository, build/pull images, run script. Get results, upload them to github. Share, discuss, analyze. The setup stage is minimized and focus shifts to results analysis.
5. Use various instances of the same framework (version/compilation flags ...) transparently. They all differ by a container identifier. Everything else remains the same.  
6. Throw away SW installation routines that may take days. Git rid of dependency problems. Applications run in sandbox environments - they will never pick the wrong library at runtime - save hours of debugging for something usefull.
7. Push new version of your container to your registry. It then immediately becomes available to all hosts in a cluster.

This document covers the following topics:
* [Requirements to run experiments in containers](#requirements-to-run-experiments-in-containers)
* [Installing Docker on RedHat OS](#installing-docker-on-redhat-os)
* [Running Docker as a non root user](#running-docker-as-a-non-root-user)
* [Setting up proxy server](#setting-up-proxy-server)
* [Installing NVIDIA Docker](#installing-nvidia-docker)
* [What can go wrong with docker or nvidia-docker](#what-can-go-wrong-with-docker-or-nvidia-docker)
* [Docker networking](#docker-networking)
* [Docker performance](#docker-performance)
* [Docker performance metrics](#docker-performance-metrics)
* [Building images vs pulling them from central repository](#building-images-vs-pulling-them-from-central-repository)
* [Building/pulling images](#buildingpulling-images)
* [Images that are run by standard benchmarking scripts](#images-that-are-run-by-standard-benchmarking-scripts)
* [Quick cheatsheet](#quick-cheatsheet)
* [Best practices for writing Dockerfiles](#best-practices-for-writing-dockerfiles)

### Requirements to run experiments in containers
Mandatory:
* `docker`/`docker-engine`
* `nvidia-docker`

Optional:
* `python` (some code that analyzes results will not work)
* `awk` (some code that analyzes results will not work)

The mandatory requirement to run CPU based benchmarks/experiments is docker. There are several packages that provide docker, for instance, `docker-engine` and `docker.io`. For CPU only machines, they both can work. However, if GPU tests are also to be run, `docker-engine` must be used.

Docker installation guide can be found [here](https://docs.docker.com/engine/getstarted/step_one/). This document contains step by step installation guides for some operating systems.

For GPU-based experiments, `nvidia-docker` must be installed.

### Installing Docker on RedHat OS
Whenever you try to install latest version of Docker CE, you get the following message:

```
WARNING: redhat is now officially only supported by Docker EE
         Check https://store.docker.com for information on Docker EE
```

However, it's possible to install Docker CE for CentOS. This is of course an unsupported configuration. These are the steps to install Docker CE taken from [here](https://stackoverflow.com/questions/42981114/install-docker-ce-17-03-on-rhel7):

Set up the Docker CE repository on RHEL:
```
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum makecache fast
```
Install the latest version of Docker CE on RHEL (if you get errors, scroll down for possible solutions):
```
sudo yum -y install docker-ce
```
If you get error similar to this one:
```
Error: Package: docker-ce-17.06.0.ce-1.el7.centos.x86_64 (docker-ce-stable)
       Requires: container-selinux >= 2.9
```
You can try to do the following:
```
wget http://mirror.centos.org/centos/7/extras/x86_64/Packages/container-selinux-2.10-2.el7.noarch.rpm
sudo yum install policycoreutils-python
sudo rpm -ivh container-selinux-2.10-2.el7.noarch.rpm
sudo yum install docker-ce
```

Start Docker:
```
sudo systemctl start docker
```
### Running Docker as a non root user

Add yourself to a docker group to be able to run containers as a non-root user

```
sudo groupadd docker             # Add the docker group if it doesn't already exist.
sudo gpasswd -a ${USER} docker   # Add the connected user "${USER}" to the docker group. Change the user name to match your preferred user.
sudo service docker restart      # Restart the Docker daemon.
newgrp docker                    # Either do a newgrp docker or log out/in to activate the changes to groups.
```

### Setting up proxy server

If you are behind a proxy server, make sure Docker is aware about it.
1. Create a systemd drop-in directory for the docker service

     ```bash
     sudo mkdir /etc/systemd/system/docker.service.d
     ```
  2. Create a file `/etc/systemd/system/docker.service.d/http-proxy.conf` with the following content (figure out your proxy servers):

     ```text
     [Service]
     Environment="HTTP_PROXY=${HTTP_PROXY_SERVER_HOST}:${HTTP_PROXY_SERVER_PORT}"
     Environment="HTTPS_PROXY=${HTTPS_PROXY_SERVER_HOST}:${HTTPS_PROXY_SERVER_PORT}"
     ```
  3. Flush changes

     ```bash
     sudo systemctl daemon-reload
     ```
  4. Verify that the configuration has been loaded

     ```bash
     sudo systemctl show --property Environment docker
     ```
  5. Restart Docker

     ```bash
     sudo systemctl restart docker
     ```

### Installing NVIDIA Docker

Install nvidia-docker and nvidia-docker-plugin (check for latest version on GitHub). Detailed NVIDIA's docker installation guide is [here](https://github.com/NVIDIA/nvidia-docker). Check it for specific details for your OS. This is the example that works in RedHat and CentOS:

```bash
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker-1.0.1-1.x86_64.rpm
sudo rpm -i /tmp/nvidia-docker*.rpm && rm /tmp/nvidia-docker*.rpm
sudo systemctl start nvidia-docker
```

### What can go wrong with docker or nvidia-docker
There are several things related to running docker containers that may go wrong. Before running tests or benchmarks, make sure you can run _docker_ and/or _nvidia-docker_ containers by invoking following commands (assuming you have built some `image_name:tag` image):
```
docker run -ti image_name:tag /bin/bash
nvidia-docker run -ti image_name:tag /bin/bash
```
If docker could successfully start containers, everything is ok. These are the typical issues that you may face:
* `You cannot run containers as a non root user`. Run as root (not recommended), google for solution, or use the above mentioned sequence of commands to allow other users running containers.

* `NVIDIA docker plugin is not running`. You may need to start this plugin manually:
```
sudo nohup nvidia-docker-plugin &
```
* `Get https://registry-1.docker.io/v2/: x509: certificate signed by unknown authority` error during image building process. Make sure Docker is aware about your proxy server. See this [section](#setting-up-proxy-server).
* Make sure NVIDIA's known [limitations](https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker-plugin#known-limitations) are not applicable to you. In general, this sometimes happens. The simplest case is to restart nvidia docker plugin with `-d` parameter specifying non standard location (use `nvidia-docker-plugin --help` for more details).

* Anyway, a good way to find out what's wrong with nvidia docker plugin, is to stop it if running and then start it as a foreground process `nvidia-docker-plugin` in separate terminal. Then, in some other window run `nvidia-docker ...` and study the output. Some useful links to start solving issues:
  * [NVIDIA Docker plugin](https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker-plugin).
  * [How do I change the Docker image installation directory?](https://forums.docker.com/t/how-do-i-change-the-docker-image-installation-directory/1169)

#### Issue 1: `/var` is located on separate hard drive.
This relates to known limitations (see above). If it is the first time you install docker and THERE ARE NO IMAGES that you use (check `docker images`), the best and easiest way to solve it is like this (RHEL example):

1. Stop docker daemon: `sudo service docker stop`
2. Remove folder /var/lib/docker: `rm -rf /var/lib/docker`
3. Create new folder on a appropriate drive, for instance, `sudo mkdir -o /opt/docker`
4. Create a symlink from /opt/docker to /var/lib/docker
5. Start docker daemon: `sudo service docker start`

NVIDIA's docker must also be run in this case in a special way: `sudo nvidia-docker-plugin -d /opt/nvidia-docker` i.e. `sudo nohup nvidia-docker-plugin -d /opt/nvidia-docker &`.

#### Issue 2: undefined reference to functions from `nvml` library.
I faced this error couple of times compiling NVIDIA Caffe. You may get errors something like these:
```bash
[ 87%] Linking CXX executable cifar10/convert_cifar_data
../lib/libcaffe-nv.so.0.16.1: undefined reference to `nvmlDeviceSetCpuAffinity'
../lib/libcaffe-nv.so.0.16.1: undefined reference to `nvmlInit_v2'
../lib/libcaffe-nv.so.0.16.1: undefined reference to `nvmlDeviceGetCount_v2'
../lib/libcaffe-nv.so.0.16.1: undefined reference to `nvmlShutdown'
../lib/libcaffe-nv.so.0.16.1: undefined reference to `nvmlDeviceGetHandleByIndex_v2'
collect2: error: ld returned 1 exit status
```
In theory, this should not be a problem since base images provided by NVIDIA define `LIBRARY_PATH` variable containing path to a
folder containing stub libraries (/usr/local/cuda/lib64/stubs). See, for instance, this [Dockerfile](https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/8.0/devel/Dockerfile). GCC should be able to use `LIBRARY_PATH` variable. If this
error occurs, edit the corresponding docker file. NVIDIA Caffe Dockerfile contains example command line that you can use to fix this
error.

### Docker networking
Every host can run docker containers and they can communicate with each other. We provide a short overview of how we implement multi-host Docker networking [here](../docs/docker_networking.md). That document also presents the tool `test_bandwidth.sh` that measures ethernet and infiniband bandwidth achievable between two processes running on host or inside docker containers (spoiler: it demonstrates that certain docker networking mechanisms do not introduce overhead and demonstrate similar performance as in a non-containerized environment, it also demonstrates how to use InfiniBand inside Docker containers).

See `Docker performance` section for more details, in particular, in the referenced IBM's [report](http://domino.research.ibm.com/library/cyberdig.nsf/papers/0929052195DD819C85257D2300681E7B/$File/rc25482.pdf) they state that port mapping may introduce a minor latency Solution - use host network instead (`--net=host`).

### Docker performance
In section docker networking, we demonstrate that certain implementations demonstrate same performance as non-containerized applications.

Ching-Hsiang Chu from Ohio state university did extensive testing of Docker workloads, slide deck is available [here](http://web.cse.ohio-state.edu/~panda.2/5194/slides/9e_9h_virtualization.pdf), slides 5-15. Their results show that containers impose almost no overhead on CPU and memory usage, they only impact I/O and OS interaction. I assume that the same should be true for GPUs.

Another greate source of information and reference is [this stackoverflow thread](https://stackoverflow.com/questions/21889053/what-is-the-runtime-performance-cost-of-a-docker-container).
Several usefull references from that page:

* [Linux Containers - NextGen Virtualization for Cloud](https://www.youtube.com/watch?v=a4oOAVhNLjU)
* [An Updated Performance Comparison of Virtual Machines and Linux Containers](http://domino.research.ibm.com/library/cyberdig.nsf/papers/0929052195DD819C85257D2300681E7B/$File/rc25482.pdf).
Authors compare bare metal, KVM and Docker containers. `The general result is that Docker is nearly identical to Native performance and faster than KVM in every category.`.
Also, see discussion on stackoverflow for critical review of this report.

### Docker performance metrics

* [Analyzes resource usage and performance characteristics of running containers](https://github.com/google/cadvisor)
* [Gathering LXC and Docker containers metrics](https://blog.docker.com/2013/10/gathering-lxc-docker-containers-metrics/)

### Building images vs pulling them from central repository
For better performance (or for additional functionality) you want to build your own image so that the code takes advantages of your hardware. This may not be that critical for GPU workloads but definitely may speed up CPU ones. TensorFlow issues warnings if it detects that the compiled version does not take advantage of advanced hardware capabilities such as, for instance, AVX instructions:
```
The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
```
We strongly reccommend building images on every machine for CPU benchmarks. In case of GPU experiments, if some computationally expensive code runs on CPUs (like for instance, data preprocessing, or components in TensorFlow that does not implement operations on GPUS) it is also advantageous to use compiled version.


### Building/pulling images
* `docker pull <image>` - Get the latest version of the image. If previous version is installed, it will be upgraded.
* `build.sh framework/device`  -  Build images locally. Here, we assume that you only want to build images that are provided with this project. Every framework, or its variation, has its own folder: `bvlc_caffe`, `intel_caffe`, `tensorflow` etc. The `device` tag represents framework version - either CPU or GPU based. To list available images, run `build.sh` without arguments. Every image built in this manner will have the following name: `hpe/framework:device`.


### Images that are run by standard benchmarking scripts
```
cat /usr/local/cuda/version.txt
find /usr/lib -name *cudnn*
find /usr/lib -name *nccl*

#TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"
#Caffe2
cat /opt/caffe2/VERSION_NUMBER
#MXNET
cat /opt/mxnet/NEWS.md
#PyTorch
cat /opt/pytorch/pytorch/setup.py | grep 'version ='
Caffe
cat /opt/caffe/CMakeLists.txt | grep CAFFE_TARGET_VERSION
```

#### Recommended images from NVIDIA GPU Cloud
They may not be the latest images but they have been tested with DLBS backend.


| Image ID                            | Accelerator Software                  | Framework    | Version |Target  Device  |
|-------------------------------------|---------------------------------------|--------------|---------|----------------|
| nvcr.io/nvidia/tensorflow:18.04-py3 | CUDA-9.0.333/cudnn-7.1.1/nccl-2.1.15  | TensorFlow   | 1.7.0   | GPU            |
| nvcr.io/nvidia/caffe2:18.05-py2     | CUDA-9.0.333/cudnn-7.1.2/nccl-2.1.15  | Caffe2       | 0.8.1   | GPU            |
| nvcr.io/nvidia/mxnet:18.05-py2      | CUDA-9.0.333/cudnn-7.1.2/nccl-2.1.15  | MXNET        | 1.1.0   | GPU            |
| nvcr.io/nvidia/pytorch:18.05-py3    | CUDA-9.0.333/cudnn-7.1.2/nccl-2.1.15  | PyTorch      | 0.4.0a0 | GPU            |
| nvcr.io/nvidia/caffe:18.05-py2      | CUDA-9.0.333/cudnn-7.1.2/nccl-2.1.15  | NVIDIA Caffe | 0.17.0  | GPU            |


#### Reference (baseline) images built with DLBS

| Image ID                            | Base Image                                | Accelerator Software                  | Framework    | Version |Target  Device  |
|-------------------------------------|-------------------------------------------|---------------------------------------|--------------|---------|----------------|
| hpe/benchmarks:ethernet             | nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04  | N/A                                   | N/A          |         | CPU            |
| hpe/benchmarks:infiniband           | nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04  | N/A                                   | N/A          |         | CPU            |
| hpe/bvlc_caffe:cuda9-cudnn7         | nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04  | CUDA-9.0.176/cudnn-7.1.3/nccl-2.1.15  | BVLC Caffe   | 1.0.0   | GPU            |
| hpe/nvidia_caffe:cuda9-cudnn7       | nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04  | CUDA-9.0.176/cudnn-7.1.3/nccl-2.1.15  | NVIDIA Caffe | 0.16.1  | GPU            |
| hpe/intel_caffe:cpu                 | ubuntu:16.04                              | MKL2017                               | Intel Caffe  | master  | CPU            |
| hpe/caffe2:cuda9-cudnn7             | nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04  | CUDA-9.0.176/cudnn-7.1.3/nccl-2.1.15  | Caffe2       | 0.8.1   | GPU            |
| hpe/mxnet:cuda9-cudnn7              | nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04  | CUDA-9.0.176/cudnn-7.1.3/nccl-2.1.15  | mxnet        | 1.0.0   | GPU            |
| hpe/tensorflow:cuda9-cudnn7         | nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04  | CUDA-9.0.176/cudnn-7.1.3/nccl-2.1.15  | TensorFlow   | 1.6.0   | GPU            |
| hpe/pytorch:cuda9-cudnn7            | nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04  | CUDA-9.0.176/cudnn-7.1.3/nccl-2.1.15  | PyTorch      | 0.4.0a0 | GPU            |

We are planning to change naming convention for reference images that will follow the pattern: `dlbs/framework:YY.MM`.



### Quick cheatsheet
* `docker pull <image>` - Get the latest version of the image. If previous version is installed, it will be upgraded.
* `docker run -it <image> <cmd>` - Run this image and execute the specified command. If called from scripts, you probably do not want to use flag `-t`.

### Best practices for writing Dockerfiles

There are several great sources with best practices on how to write Dockerfiles. Several of those sources are these:

* [Best practices for writing Dockerfiles](https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices/)
* [Best Practices for working with Dockerfiles](https://medium.com/@nagarwal/best-practices-for-working-with-dockerfiles-fb2d22b78186)
* [Docker Best Practices](https://github.com/FuriKuri/docker-best-practices)
* [How to write excellent Dockerfiles](https://rock-it.pl/how-to-write-excellent-dockerfiles/)

Another advice that can help in minimzing space occupied by image layers.

1. `Scenario`: Install application/driver/library from a very large file stored on your hard drive. `Why is it a problem?`: Normally, you need to copy (`COPY` command) this file inside an image during a build process. This will create a layer with that file even though you will not need it and after installation you will delete it. `Solution`: Run `python -m SimpleHTTPServer` from the folder on your HDD where you store that file. It will enable http access to that file over the network. You can use normal `RUN` command in Dockerfile to download it (wget) and then delete it. Since it is all done inside a single `RUN` command, there will be no trace of this large file left in image layers.
