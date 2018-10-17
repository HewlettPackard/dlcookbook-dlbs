# Singularity common/

The common directory is where, as the name implies, files and scripts common to building and managing all of DLBS Singularity images.
The building of Singularity images requires root privilege at the host level typically using sudo.  This is dangerous and should be limited to a designated,
somewhat isolated machine or building should be restricted to a trusted few.

## Scripts:


*  build.py - this is a convenience script to run a build with a Singularity recipe.  You must have sudo for singularity or root to run it.
```
    usage: build.py [-h] --recipe RECIPE [--macros MACROS] [--image IMAGE]
    
    Build a Singularity image from a recipe template file.
    
    optional arguments:
      -h, --help            show this help message and exit
      --recipe RECIPE, -r RECIPE
                            Singularity recipe file or tempate file to process.
      --macros MACROS, -m MACROS
                            Optional JSON file containing macro substitutions.
      --image IMAGE, -i IMAGE
                            Name of Singularity image to create.
```
If -i is not specified it will default to "image.img". 

*  common_macros_10.json - This file is used to define values that will replace the placeholder tags in the Singularity recipe file. In the recipe file, the tags are
indicate by !{indentifier}.  In the macro file the substitution is defined by "identifier" : "text to substitute". The file must past the JSON syntax checker which means
basically that the identifier and values are enclosed in double quotes and that they are comma separated except for that last element in a bracketed block
(unlike Python).  

This is where you can specify software versions for various libraries and other software such as
Mellanox OFED and HPCX SDK.  Whenever a version changes, update this file rather than change the Singularity recipes.
The export statements contained in the file are used in multiple stages: 1) The setup section to copy in the prerequisite files (see below for the
list of files that must be obtained before building images). The contents of the file are also appended to /.singularity.d/env/90environment.sh
which will be sourced, and the variables added to the environment, every time the Singularity image is instantiated.

The point of doing this is to remove version dependencies from the Singularity recipes themselves to the extent possible.

## Common Singularity recipe files

These are Singularity recipe files to help with managing the Singularity build process - mostly cleaning up the detritus.
All of these recipe files since they run with root privilege are dangerous.
One note:  The script, build.sh, will create an image called "image.img" if the -n option is not specified. For the following recipes, delete this file.

* Singularity.chown - When a build fails, it can leave root own stuff.  This uses the sudo permission to run chown. This is d
* Singularity.cleanup -  The Singularity build process tends to leave root owned directories in /tmp. This recipe file will do the cleanup.
* Singularity.common - This recipe builds a base image common to the framework Recipes from a base Ubuntu/Nvidia Cuda Docker image.
  This image adds libraries, Mellanox HCX SDX and OpenMPI and some utilities to the original image.

##Other files that are included.
*	jupyter_notebook_config.py - this gets incorpated into the images. Edit to change startup options for Jupyter notebooks.

These will be changed as NVidia changes them. They are under the Apache OSS license. Note they these are unmodified. There are also modified versions for use in DLBS
in dlbs/python/ (mostly additional logging output).

## Files that you must download yourself.

To build with Singularity.common and the framework recipes, the following archives, packages are required. Some, with obvious names, are only required by
certain specific frameworks. Some of these require an NVidia account to download.

The recipe files in this directory tree expect the files to be in the /path/to/dlbs/singularity/common directory. Note that the versions are rapidly changing and the macro file must be updated to account for any changes.

*	 Anaconda3-!{ANACONDA3_VERSION}-!{ANACONDA3_ARCH}.sh - Download the latest Linux 64-bit Python 3 from anaconda.com.
*    [Mellanox HPCX SDK. Includes OpenMPI.  hpcx-!{HPCX_VERSION}-gcc-MLNX_OFED_LINUX-!{OFED_VERSION}-ubuntu!{UBUNTU_VERSION}-x86_64.tbz] (http://content.mellanox.com/hpc/hpc-x/v2.0/hpcx-!{HPCX_VERSION}-gcc-MLNX_OFED_LINUX-!{OFED_VERSION}-ubuntu!{UBUNTU_VERSION}-x86_64.tbz) You can uncomment the wget in the recipe file but this is more efficient.
*    [Mellanox OFED drivers MLNX_OFED_LINUX-!{OFED_VERSION}-ubuntu!{UBUNTU_VERSION}-x86_64.tgz] ( http://www.mellanox.com/page/mlnx_ofed_eula?mtag=linux_sw_drivers&mrequest=downloads&mtype=ofed&mver=MLNX_OFED-!{OFED_VERSION}&mname=MLNX_OFED_LINUX-!{OFED_VERSION}-ubuntu!{UBUNTU_VERSION}-x86_64.tgz) Requires clicking on acceptance of EULA. So can't wget from the recipe file unless you do something crafty.  This will of course change as version changes.
*	 nccl_!{"NCCL_VERSION"}+!{NCCL_ARCH}.txz - Nvidia NCCL collective communications library for CUDA.
*    nv-peer-memory:  nvidia-peer-memory_1.0-5_all.deb nvidia-peer-memory-dkms_1.0-5_all.deb. Install on the host(s) with instructions from: https://github.com/Mellanox/nv_peer_memory.  Alternatively you can do it all in the recipe file by uploading the original tar file, nvidia-peer-memory_1.0.5.tar.gz.
*	 pycuda-2018.1.tar.gz - It proved difficult do a pip install so download the archive and build: See: https://wiki.tiker.net/PyCuda/Installation/Linux
Up through dpkg-buildpackage -us -uc. This will create the .deb packages above.

## Building the image
```
./build.sh -r Singularity.common -m common_macros_10.json -i nvidia_cuda_10.0_cudnn_7_ubuntu16.04_common.img
# Move the image to a central location.  
# The TensorFlow recipes assume it will be in /var/lib/SingularityImage
# but you can change that by editing the recipe files.
mv nvidia_cuda_10.0_cudnn_7_ubuntu16.04_common.img /var/lib/SingularityImages
#cleanup
./build.sh -r Singularity.cleanup
rm image.img
```
