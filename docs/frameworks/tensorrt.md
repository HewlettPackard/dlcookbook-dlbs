# __TensorRT__

In order to be able to run TensorRT benchmarks, you need to sign up and request access
to TensorRT inference engine. See this [page](https://developer.nvidia.com/tensorrt)
for more details.

We currently support 2.0.2 version. Other versions may work as well, but I did not test
them. TensorRT can be benchmarked in host environment or in docker container.
For host benchmarks, install the TensorRT package. To build docker container,
copy this file `nv-tensorrt-repo-ubuntu1604-7-ea-cuda8.0_2.0.2-1_amd64.deb` into
[tensorrt/cuda8-cudnn6](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/tensorrt/cuda8-cudnn6)
folder and build an image:
```bash
build.sh tensorrt/cuda8-cudnn6
```

## Standalone run

TensoRT benchmark project - a backend for DLBS - is located in
[src/tensorrt](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/src/tensorrt)
folder.

More details on this project are in readme file
[here](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/src/tensorrt/README.md).

## Models
Same as Caffe. Details are [here](/frameworks/caffe.md?id=models)

## Adding new model
Same as Caffe. Details are [here](/frameworks/caffe.md?id=adding-new-model)

## Commonly used configuration parameters

### __tensorrt.docker.image__ =`hpe/tensorrt:cuda8-cudnn6`
The name of a docker image to use for TensorRT.
### __tensorrt.host.path__ =`${DLBS_ROOT}/src/tensorrt/build`
Path to a tensorrt executable in case of bare metal run.

## Other parameters

### __tensorrt.profile__ =`false`
If true, per layer statistics are measured.
### __tensorrt.input__ =`data`
Name of an input data tensor (data).
### __tensorrt.output__ =`prob`
Name of an output tensor (prob).
### __tensorrt.host.libpath__ =`""`
Basically, it's a LD_LIBRARY_PATH for TensorRT in case of a bare metal run
(should be empty).

## Internal parameters

### __tensorrt.launcher__ =`${DLBS_ROOT}/scripts/launchers/tensorrt.sh`
Path to script that launches TensorRT benchmarks.
### __tensorrt.args__ =...
Command line arguments that launcher uses to launch TensorRT.
### __tensorrt.model_file__ =`${exp.id}.model.prototxt`
Caffe's prototxt inference (deploy) model file.
### __tensorrt.model_dir__ =`$('${DLBS_ROOT}/models/${exp.model}' if '${exp.env}' == 'host' else '/workspace/model')$`
Directory where Caffe's model file is located. Different for host/docker
benchmarks.
### __tensorrt.docker.args__ =...
In case if containerized benchmarks, this are the docker parameters.
