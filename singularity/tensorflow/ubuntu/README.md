# Singularity tensorflow/ubuntu/

This directory contains the recipe files for building a TensorFLow Singularity image.
Due to the fact that an error at any time will mean that the process has to start from the beginning, the build process here-in is done in stages
with intermediate images. Furthermore, a base TensorFlow image is created and then "sub-classed" with customized additions.

## Recipes:
*  The macro file, curently tensorflow_macros_10.json.  When versions change update the macro file.  This file is parsed by build.py and all
tags, !{ident}, are replaced. You should create different versions of the macro file for different builds, such as for different versions of TensorFlow or CUDA.
*  Singularity.tensorflow.prelim - This recipe file takes the common image built in dlbs/singularity/common as the base image.  It is expected that the base image is nameed !{CUDA_VERSION}_cudnn_!{CUDNN_MAJOR_VERSION}_ubuntu!{UBUNTU_VERSION}_common.img, where the tags are defined in the macro file.
It installs the absl Python package that TensorFlow requires and builds and installs the Google Bazel tool and a few other things.
The version of Bazel is specified in the macro file (BAZEL_VERSION of course).

To build the image run:

```
cd /path/to/dlbs/singularity/tensorflow/ubuntu
../../build.sh --recipe Singularity.prelim --macros tensorflow_macros_10.json --image tensorflow-1.11.0-prelim.img 2>&1 log |tee   # I like to see whats going on but also have a log file.

```
*  Singularity.tensorflow.base - This builds the actual TensorFlow image from source. As of this writing, the pre-built binary from the github 
repo does not support the latest CUDA version, necessitating building from source.

This recipe file also installs Horovod using the OpenMPI installation from the common image built previosly. We don't build with
TF_NEED_VERBS for gRPC or Distributed TensorFlow since Horovod is our preferred parallelization method. You can change that by setting
TF_NEED_VERBS=1 in the recipe file.

If TF_NEED_TENSORRT is set to 1 (line 45), NVidia tensorrt will be installed. You will have to register on the NVidia developer site and download
the .deb file. The version is currently hardwired is set in the macro file. If Tensorrt is used uncomment lines 55 - 67 in Singularity.tensorflow.base. The repo
ufortunately uses a .whl (Python wheel) that will only install into Python 3.5. So these lines copy and rename into the Anaconda Python tree. Version 5 does not have this
problem.

To build the TensorFlow image:
```
../../build.sh -r Singularity.base -m tensorflow_macros_10.json  -i tensorflow-1.11.0-base.img 2>&1 log |tee
```
Here, you can see the short form of the options.

If you don't plan to build with the additional goodies you can name the image tensorflow-1.11.0.img and move it the /var/SingularityImages,
or whereever its final destination may be.

*  Singularity.tensorflow.additional - This recipe file will build additional capability in to the base TensorFlow image. Currently the following
are capabilities are added:

..*  Standalone Keras - TensorFlow has a Keras module but as not everything is implemented there are some minor API incompatibilies, the separate
Keras package is installed. Mostly for comparison and debugging.

..* TensorFlow Hub - This package provides the capability to download pre-trained TensorFlow models for transfer learning.

..* OpenNMT-tf - Open Neural Machine Translation.  
current tg.

..* tensor2tensor - an additional library for sequential learning from the TensorFlow github site.


*  Singularity.tensorflow.geo
..* This has some additional GIS related installation.
