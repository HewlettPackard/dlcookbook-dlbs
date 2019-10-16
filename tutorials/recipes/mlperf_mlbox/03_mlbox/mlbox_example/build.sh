#!/bin/bash

# User-provided docker image name
img_name="dlbs/ngc-tf:18.07-mlbox"

# Runtime build arguments
args=""
[[ -n "${http_proxy}" ]] && args="${args} --build-arg http_proxy=${http_proxy}"
[[ -n "${https_proxy}" ]] && args="${args} --build-arg https_proxy=${https_proxy}"
docker build -t ${img_name} ${args} ./mlbox
