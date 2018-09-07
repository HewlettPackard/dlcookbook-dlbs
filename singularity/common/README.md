# Singularity common/

The common directory is where, as the name implies, files and scripts common to building and managing all of DLBS Singularity images.
The building of Singularity images requires root privlege at the host level typically using sudo.  This is dangerous and should be limited to a designated,
somewhat isolated machine or building should be restricted to a trusted few.

## Scripts:


build.sh - this is a convenience script to run a build with a Singularity recipe.  You must have sudo for singularity or root to run it.
	>  build.sh -r <recipe file> [-n <output image file>]

If -n is not specified it will default to "image.img". 

## Common Singularity recipe files

These are Singularity recipe files to help with managing the Singularity build process - mostly cleaning up the detritus.
All of these recipe files since they run with root privilege are dangerous.
One note:  The script, build.sh, will create an image called "image.img" if the -n option is not specified. For the following recipes, delete this file.

* Singularity.chown - When a build fails, it can leave root own stuff.  This uses the sudo permission to run chown. This is d
* Singularity.cleanup -  The Singularity build process tends to leave root owned directories in /tmp. This recipe file will do the cleanup.
* Singularity.common - This recipe builds a base image common to the framework Recipes from a base Ubuntu/Nvidia Cuda Docker image,
  currently nvidia_cuda_9.2_cudnn_7_ubuntu16.04.  This image adds libraries, Mellanox HCX SDX and OpenMPI and some utilities to the original image.

##Other files that are included.
*	jupyter_notebook_config.py - this gets incorpated into the images. Edit to change startup options for Jupyter notebooks.
*	nvidia-mxnet-examples.tgz - mxnet examples extracted from the NGC mxnet Docker image.  Only for mxnet.
*	nvidia-pytorch-examples.tgz - pytorch examples extracted from the NGC pytorch Docker image.  Only for pytorch.
*	nvidia-tensorflow-examples.tgz - tensorflow examples extracted from the NGC tensorflow Docker image.  Only for tensorflow.
*	nvidia-caffe2-examples.tgz -  caffe2 examples extracted from the NGC caffe2 Docker image.  Only for caffe2.
*	nvidia-caffe2-binaries.tgz - C source files for some Caffe2 utilities.

These will be changed as NVidia changes them. They are under the Apache OSS license. Note they these are unmodified. There are also modified versions for use in DLBS
in dlbs/python/ (mostly additional logging output).

## Files that you must download yourself.

To build with Singularity.common and the framework recipes, the following archives, packages are required. Some, with obvious names, are only required by
certain specific frameworks. Some of these require an NVidia account to download.

*	 hpcx-v2.0.0-gcc-MLNX_OFED_LINUX-4.2-1.0.0.0-ubuntu16.04-x86_64.tbz - Mellanox HCX SDK. Containes OpenMPI with IB support.
*	 MLNX_OFED_LINUX-4.2-1.2.0.0-ubuntu16.04-x86_64.tgz - Mellanox OFED drivers and libraries. The Singularity image requires the libraries. 
     This will of course change as version changes.
*	 nccl_2.2.13-1+cuda9.2_x86_64.txz - Nvidia NCCL collective communications library for CUDA 9.2.  Only if you are building the CUDA 9.2 image.
     This will of course change as version changes.
*	 nv-tensorrt-repo-ubuntu1604-cuda9.2-ga-trt4.0.1.6-20180612_1-1_amd64.deb - Nvidia TensorRT.  This requires an Nvidia developer account.
*	 pycuda-2018.1.tar.gz - It proved difficult do a pip install so download the archive and build: See: https://wiki.tiker.net/PyCuda/Installation/Linux

## Images to build before building frameworks.

### Pull and convert Base Nvidia CUDA and Ubuntu image.

Edit the first line Singularity.common to specify the desired Docker image (CUDA version and Ubuntu release).
./build.sh -r Singularity.common -n nvidia_cuda_9.2_cudnn_7_ubuntu16.04.img
*	 nvidia-common-cuda9.2-cudnn7.1.img
