# Singularity tensorflow/ubuntu/

This directory contains the recipe files for building a TensorFLow Singularity image.
Due to the fact that an error at any time will mean that the process has to start from the beginning, the build process here-in is done in stages
with intermediate images. Furthermore, a base TensorFlow image is created and then "sub-classed" with customized additions.

## Recipes:
*  Singularity.tensorflow.prelim - This recipe file takes the common image built in dlbs/singularity/common and currently named 
nvidia_cuda_9.2_cudnn_7_ubuntu16.04_common.img as the base image. 
It installs the absl Python package that TensorFlow requires and builds and installs the Google Bazel tool and a few other things.
The version of Bazel is currently hard-wired into the Recipe file and in this version is set to 17.2. As new versions of the install become
available change line 16:
``` 
 15 %post
 16     export BAZEL_VERSION=**0.17.2**
``` 
Just for accountability, as versions of CUDA, cuDNN and TensorFlow change, update the %labels, and %help section test where they occur. For
Singularity.tensorflow.prelim not make the changes won't have any affect other than providing misleading labels and help messages, but the image
will still work.
To build the image run:

```
cd /path/to/dlbs/singularity/tensorflow/ubuntu
../../build.sh -r Singularity.prelim -n tensorflow-1.11.0-prelim.img 2>&1 log |tee   # I like to see whats going on but also have a log file.

```
*  Singularity.tensorflow.base - This builds the actual TensorFlow image from source. As of this writing, the pre-built binary from the github 
repo does not support the latest CUDA version, necessitating building from source.

When the TensorFlow versions change, edit this recipe file and change all occurrences of the TensorFlow version, currently 1.11.0 to
the new version.  If this is not done, help and label messages will be incorrect.  Same for the CUDA and cuDNN versions.  The changes are informational only
the recipe downloads the latest release version itself.

This recipe file also installs Horovod using the OpenMPI installation from the common image built previosly. We don't build with
TF_NEED_VERBS for gRPC or Distributed TensorFlow since Horovod is our preferred parallelization method. You can change that by setting
TF_NEED_VERBS=1 in the recipe file.

If TF_NEED_TENSORRT is set to 1 (line 45), NVidia tensorrt will be installed. You will have to register on the NVidia developer site and download
the .deb file. The version is currently hardwired into this recipe file at line 18 and should be changed as versions change.

```
18     export TRT_VERSION=ubuntu1604-cuda9.2-ga-trt4.0.1.6-20180612_1-1_amd64
```
The code to install TensorRT has to do some crude manipulation to get it to work with Python 3.6. This probably won't be necessary with the next version and
lines 68-74 would then be removed.

To build the TensorFlow image:
```
../../build.sh -r Singularity.base -n tensorflow-1.11.0-base.img 2>&1 log |tee
```
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
