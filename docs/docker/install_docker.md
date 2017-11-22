# __Installing docker__

This document covers the following topics:
* [Installing Docker](#installing-docker)
* [Installing Docker on RedHat OS](#installing-docker-on-redhat-os)
* [Running Docker as a non root user](#running-docker-as-a-non-root-user)
* [Setting up proxy server](#setting-up-proxy-server)
* [Installing NVIDIA Docker](#installing-nvidia-docker)
* [What can go wrong with docker or nvidia-docker](#what-can-go-wrong-with-docker-or-nvidia-docker)
* [Docker networking](#docker-networking)
* [Docker performance](#docker-performance)
* [Docker performance metrics](#docker-performance-metrics)
* [Building images vs pulling them from central repository](#building-images-vs-pulling-them-from-central-repository)
* [Best practices for writing Dockerfiles](#best-practices-for-writing-dockerfiles)

## Installing Docker
Docker installation guide can be found [here](https://docs.docker.com/engine/getstarted/step_one/). This document contains step by step installation guides for some operating systems.
For GPU-based experiments, `nvidia-docker` must be installed.

## Installing Docker on RedHat OS
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
## Running Docker as a non root user

Add yourself to a docker group to be able to run containers as a non-root user

```
sudo groupadd docker             # Add the docker group if it doesn't already exist.
sudo gpasswd -a ${USER} docker   # Add the connected user "${USER}" to the docker group. Change the user name to match your preferred user.
sudo service docker restart      # Restart the Docker daemon.
newgrp docker                    # Either do a newgrp docker or log out/in to activate the changes to groups.
```

## Setting up proxy server

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

## Installing NVIDIA Docker

Install nvidia-docker and nvidia-docker-plugin (check for latest version on GitHub). Detailed NVIDIA's docker installation guide is [here](https://github.com/NVIDIA/nvidia-docker). Check it for specific details for your OS. This is the example that works in RedHat and CentOS:

```bash
wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker-1.0.1-1.x86_64.rpm
sudo rpm -i /tmp/nvidia-docker*.rpm && rm /tmp/nvidia-docker*.rpm
sudo systemctl start nvidia-docker
```

## What can go wrong with docker or nvidia-docker
There are several things related to running docker containers that may go wrong. Before running tests or benchmarks, make sure you can run _docker_ and/or _nvidia-docker_ containers by invoking following commands (assuming you have built some `image_name:tag` image):
```bash
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

Anyway, a good way to find out what's wrong with nvidia docker plugin, is to stop it if running and then start it as a foreground process `nvidia-docker-plugin` in separate terminal. Then, in some other window run `nvidia-docker ...` and study the output. Some useful links to start solving issues:
  * [NVIDIA Docker plugin](https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker-plugin).
  * [How do I change the Docker image installation directory?](https://forums.docker.com/t/how-do-i-change-the-docker-image-installation-directory/1169)

### Issue 1: `/var` is located on separate hard drive.
This relates to known limitations (see above). If it is the first time you install docker and THERE ARE NO IMAGES that you use (check `docker images`), the best and easiest way to solve it is like this (RHEL example):

1. Stop docker daemon: `sudo service docker stop`
2. Remove folder /var/lib/docker: `rm -rf /var/lib/docker`
3. Create new folder on a appropriate drive, for instance, `sudo mkdir -o /opt/docker`
4. Create a symlink from /opt/docker to /var/lib/docker
5. Start docker daemon: `sudo service docker start`

NVIDIA's docker must also be run in this case in a special way: `sudo nvidia-docker-plugin -d /opt/nvidia-docker` i.e. `sudo nohup nvidia-docker-plugin -d /opt/nvidia-docker &`.

### Issue 2: undefined reference to functions from `nvml` library.
I faced this error couple of times compiling NVIDIA Caffe. You may get errors something like these:
```bash
[ 87%] Linking CXX executable cifar10/convert_cifar_data
../lib/libcaffe-nv.so.0.16.1: undefined reference to 'nvmlDeviceSetCpuAffinity'
../lib/libcaffe-nv.so.0.16.1: undefined reference to 'nvmlInit_v2'
../lib/libcaffe-nv.so.0.16.1: undefined reference to 'nvmlDeviceGetCount_v2'
../lib/libcaffe-nv.so.0.16.1: undefined reference to 'nvmlShutdown'
../lib/libcaffe-nv.so.0.16.1: undefined reference to 'nvmlDeviceGetHandleByIndex_v2'
collect2: error: ld returned 1 exit status
```
In theory, this should not be a problem since base images provided by NVIDIA define `LIBRARY_PATH` variable containing path to a
folder containing stub libraries (/usr/local/cuda/lib64/stubs). See, for instance, this [Dockerfile](https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/8.0/devel/Dockerfile). GCC should be able to use `LIBRARY_PATH` variable. If this
error occurs, edit the corresponding docker file. NVIDIA Caffe Dockerfile contains example command line that you can use to fix this
error.

## Docker networking
Every host can run docker containers and they can communicate with each other. We provide a short overview of how we implement multi-host Docker networking [here](/docker/docker_network.md). That document also presents the tool `test_bandwidth.sh` that measures Ethernet and InfiniBand bandwidth achievable between two processes running on host or inside docker containers (spoiler: it demonstrates that certain docker networking mechanisms do not introduce overhead and demonstrate similar performance as in a non-containerized environment, it also demonstrates how to use InfiniBand inside Docker containers).

See `Docker performance` section for more details, in particular, in the referenced IBM's [report](http://domino.research.ibm.com/library/cyberdig.nsf/papers/0929052195DD819C85257D2300681E7B/$File/rc25482.pdf) they state that port mapping may introduce a minor latency Solution - use host network instead (`--net=host`).

## Docker performance
In section docker networking, we demonstrate that certain implementations demonstrate same performance as non-containerized applications.

Ching-Hsiang Chu from Ohio state university did extensive testing of Docker workloads, slide deck is available [here](http://web.cse.ohio-state.edu/~panda.2/5194/slides/9e_9h_virtualization.pdf), slides 5-15. Their results show that containers impose almost no overhead on CPU and memory usage, they only impact I/O and OS interaction. I assume that the same should be true for GPUs.

Another great source of information and reference is [this stackoverflow thread](https://stackoverflow.com/questions/21889053/what-is-the-runtime-performance-cost-of-a-docker-container).
Several useful references from that page:

* [Linux Containers - NextGen Virtualization for Cloud](https://www.youtube.com/watch?v=a4oOAVhNLjU)
* [An Updated Performance Comparison of Virtual Machines and Linux Containers](http://domino.research.ibm.com/library/cyberdig.nsf/papers/0929052195DD819C85257D2300681E7B/$File/rc25482.pdf).
Authors compare bare metal, KVM and Docker containers. `The general result is that Docker is nearly identical to Native performance and faster than KVM in every category.`.
Also, see discussion on stackoverflow for critical review of this report.

## Docker performance metrics

* [Analyzes resource usage and performance characteristics of running containers](https://github.com/google/cadvisor)
* [Gathering LXC and Docker containers metrics](https://blog.docker.com/2013/10/gathering-lxc-docker-containers-metrics/)

## Building images vs pulling them from central repository
For better performance (or for additional functionality) you want to build your own image so that the code takes advantages of your hardware. This may not be that critical for GPU workloads but definitely may speed up CPU ones. TensorFlow issues warnings if it detects that the compiled version does not take advantage of advanced hardware capabilities such as, for instance, AVX instructions:
```
The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
```
We strongly reccommend building images on every machine for CPU benchmarks. In case of GPU experiments, if some computationally expensive code runs on CPUs (like for instance, data preprocessing, or components in TensorFlow that does not implement operations on GPUS) it is also advantageous to use compiled version.

## Best practices for writing Dockerfiles

There are several great sources with best practices on how to write Dockerfiles. Several of those sources are these:

* [Best practices for writing Dockerfiles](https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices/)
* [Best Practices for working with Dockerfiles](https://medium.com/@nagarwal/best-practices-for-working-with-dockerfiles-fb2d22b78186)
* [Docker Best Practices](https://github.com/FuriKuri/docker-best-practices)
* [How to write excellent Dockerfiles](https://rock-it.pl/how-to-write-excellent-dockerfiles/)

Another advice that can help minimizing space occupied by image layers.

1. `Scenario`: Install application/driver/library from a very large file stored on your hard drive. `Why is it a problem?`: Normally, you need to copy (`COPY` command) this file inside an image during a build process. This will create a layer with that file even though you will not need it and after installation you will delete it. `Solution`: Run `python -m SimpleHTTPServer` from the folder on your HDD where you store that file. It will enable http access to that file over the network. You can use normal `RUN` command in Dockerfile to download it (wget) and then delete it. Since it is all done inside a single `RUN` command, there will be no trace of this large file left in image layers.
