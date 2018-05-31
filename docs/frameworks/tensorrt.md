# __TensorRT__

In order to be able to run TensorRT benchmarks, you need to sign up and request access
to TensorRT inference engine. See this [page](https://developer.nvidia.com/tensorrt)
for more details.

We currently support 3.0.4 version. Other versions may work as well, but I did not test
them. TensorRT can be benchmarked in host environment or in docker container.
For host benchmarks, install the TensorRT package. To build docker container,
copy this file `nv-tensorrt-repo-ubuntu1604-ga-cuda9.0-trt3.0.4-20180208_1-1_amd64.deb` into
[tensorrt/cuda9-cudnn7](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/docker/tensorrt/cuda9-cudnn7)
folder and build an image:
```bash
./build.sh tensorrt/cuda9-cudnn7
```

## Standalone run

TensorRT benchmark project - a backend for DLBS - is located in
[src/tensorrt](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/src/tensorrt)
folder.

More details on this project are in readme file
[here](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/src/tensorrt/README.md).

## Models
Same as Caffe. Details are [here](/frameworks/caffe.md?id=models)

## Adding new model
Same as Caffe. Details are [here](/frameworks/caffe.md?id=adding-new-model)

## Commonly used configuration parameters
#### __tensorrt.docker_image__

* __default value__ `"hpe/tensorrt:cuda9-cudnn7"`
* __description__ The name of a docker image to use for TensorRT.

#### __tensorrt.host_path__

* __default value__ `"${DLBS_ROOT}/src/tensorrt/build"`
* __description__ Path to a tensorrt executable in case of bare metal run.


## Other parameters
#### __tensorrt.args__

* __default value__ `[u'--model ${tensorrt.model_dir}/${tensorrt.model_file}', u'--batch_size ${exp.replica_batch}', u'--dtype ${exp.dtype}', u'--num_warmup_batches ${exp.num_warmup_batches}', u'--num_batches ${exp.num_batches}', u"$('--profile' if ${tensorrt.profile} is True else '')$", u'--input ${tensorrt.input}', u'--output ${tensorrt.output}']`
* __description__ Command line arguments that launcher uses to launch TensorRT.

#### __tensorrt.docker_args__

* __default value__ `[u'-i', u'--security-opt seccomp=unconfined', u'--pid=host', u'--volume=${DLBS_ROOT}/models/${exp.model}:/workspace/model', u"$('--volume=${runtime.cuda_cache}:/workspace/cuda_cache' if '${runtime.cuda_cache}' else '')$", u"$('--volume=${monitor.pid_folder}:/workspace/tmp' if ${monitor.frequency} > 0 else '')$", u'${exp.docker_args}', u'${tensorrt.docker_image}']`
* __description__ In case if containerized benchmarks, this are the docker parameters.

#### __tensorrt.env__

* __default value__ `"${runtime.EXPORT_CUDA_CACHE_PATH}"`
* __description__ Environmental variables to set for TensorRT benchmarks.

#### __tensorrt.host_libpath__

* __default value__ `""`
* __description__ Basically, it's a LD_LIBRARY_PATH for TensorRT in case of a bare metal run \(should be empty\).

#### __tensorrt.input__

* __default value__ `"data"`
* __description__ Name of an input data tensor \(data\)

#### __tensorrt.launcher__

* __default value__ `"${DLBS_ROOT}/scripts/launchers/tensorrt.sh"`
* __description__ Path to script that launches TensorRT benchmarks.

#### __tensorrt.model_dir__

* __default value__ `"$('${DLBS_ROOT}/models/${exp.model}' if ${exp.docker} is False else '/workspace/model')$"`
* __description__ Directory where Caffe's model file is located. Different for host/docker benchmarks.

#### __tensorrt.model_file__

* __default value__ `"${exp.id}.model.prototxt"`
* __description__ Caffe's prototxt inference \(deploy\) model file.

#### __tensorrt.output__

* __default value__ `"prob"`
* __description__ Name of an output tensor \(prob\)

#### __tensorrt.profile__

* __default value__ `False`
* __description__ If true, per layer statistics are measured.
