FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER sergey.serebryakov@hpe.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
	numactl   \
	iperf \
	libmlx4-1 \
	libmlx5-1 \
	libibverbs1 \
	ibverbs-utils \
	librdmacm1 \
	ibutils \
	libdapl2 \
	dapl2-utils
