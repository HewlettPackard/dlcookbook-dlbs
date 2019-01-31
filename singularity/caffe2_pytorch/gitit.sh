!/bin/bash
PYTORCH_ROOT=./pytorch
git clone --branch master --recursive https://github.com/pytorch/pytorch.git ${PYTORCH_ROOT}
cd ${PYTORCH_ROOT}
git reset --hard 3749c581b79cba49f511b19fa02c0f50fa05b250
git submodule update --init

