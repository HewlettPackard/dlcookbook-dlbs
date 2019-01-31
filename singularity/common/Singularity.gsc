Bootstrap: docker
FROM: docker://google/cloud-sdk
# This recipe downloads the CUDA 9.2 / Ubuntu 16.04 base Docker image and converts it to a Singularity image.
%post
    apt-get update
    apt-get install -y --no-install-recommends build-essential vim
    apt-get clean
    rm -rf /var/lib/apt/lists/*
%runscript
	/bin/bash
