FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER sergey.serebryakov@hpe.com

# This is a general image for GPU based Caffe2 workloads.

RUN apt-get update && apt-get install -y --no-install-recommends \
	numactl \
	iperf \
	iperf3  && \
	rm -rf /var/lib/apt/lists/*
