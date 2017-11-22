# __Parameters__

Most of the input parameters are defined in json files in the
[config](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/dlbs/configs) folder.
Common parameters are defined in `base.json` file. Framework specific parameters are defined
in corresponding json files.

Logically, all input parameters can be divided into three categories:

1. `Commonly used parameters`. These are parameters that are used in 95% or so
   benchmarking experiments. They include such parameters as batch size, model
   name, framework etc.
2. `Parameters`. These are parameters that may be used to additionally tune/vary
   benchmarking process. They include type of data (float16, int8), path to datasets,
   distributed aggregation schemas etc.
3. `Internal parameters`. Internal parameters should not normally be used. They
   are computed based on other parameters. In certain situations, they may be
   overridden though. They include such parameters as docker arguments, launchers
   specifications, system paths etc.

All parameters are described in [frameworks](frameworks/frameworks?id=commonly-used-configuration-parameters)
sections and in sections for every framework.

You can provide any parameters you want when running experiments. They will be ignored
and dumped into log files that you can read later on. This is useful for tracking
hardware / benchmark details. For instance, you may use something like `comments.description`
parameter to provide benchmark description.

Experimenter script can print parameters and their descriptions. It can also search
for key words in parameter descriptions and print out matches. See this
[file](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/dlbs/help/helper.py)
for more details:
```bash
# Show help on every parameter. Will print a lot of information.
python experimenter.py help --params

# Show help on parameter based on regexp match
python experimenter.py help --params exp.device
python experimenter.py help --params exp.*

# Perform full-text search in parameters description, case insensitive
python experimenter.py help --text cuda

# Perform full-text search in a subset of parameters that match params
python experimenter.py help --params exp.device --text batch

# Show most commonly used parameters for TensorFlow
experimenter.py help --frameworks tensorflow
```

## Commonly used parameters

### __Common parameters for all frameworks__
[`exp.framework`](/frameworks/frameworks?id=expframework-quotquot "Deep Learning framework (tensorflow,caffe2,tensorrt,mxnet,bvlc_caffe,nvidia_caffe,intel_caffe)")
[`exp.model`](/frameworks/frameworks?id=expmodel-quotquot "Neural network model (alexnet, googlenet, resnet50 ...)")
[`exp.env`](/frameworks/frameworks?id=expenv-host "Docker or host (docker, host)")
[`exp.warmup_iters`](/frameworks/frameworks?id=expwarmup_iters-2 "Number of warmup iterations")
[`exp.bench_iters`](/frameworks/frameworks?id=expbench_iters-100 "Number of benchmarking iterations")
[`exp.phase`](/frameworks/frameworks?id=expphase-training "Phase - training OR inference")
[`exp.device_batch`](/frameworks/frameworks?id=expdevice_batch-16 "Per device batch")
[`exp.gpus`](/frameworks/frameworks?id=expgpus-0 "Comma separated list of GPUs if GPU based benchmark")
[`exp.log_file`](/frameworks/frameworks?id=explog_file-expexp_pathexp39expgpus39replace39393939_expmodel_expeffective_batchlog "Benchmark log file")

### __Caffe__
BVLC Caffe
[`bvlc_caffe.host.path`](/frameworks/caffe?id=bvlc_caffehostpath-homeprojectsbvlc_caffebuildtools "Path to a BVLC Caffe executable in case of a bare metal run")
[`bvlc_caffe.host.libpath`](/frameworks/caffe?id=bvlc_caffehostlibpath- "Basically, it's a LD_LIBRARY_PATH for BVLC Caffe in case of a bare metal run.")
[`bvlc_caffe.docker.image`](/frameworks/caffe?id=bvlc_caffedockerimage-hpebvlc_caffecuda9-cudnn7 "The name of a docker image to use for BVLC Caffe.")

NVIDIA Caffe
[`nvidia_caffe.host.path`](/frameworks/caffe?id=nvidia_caffehostpath-homeprojectsnvidia_caffebuildtools "Path to a NVIDIA Caffe executable in case of a bare metal run.")
[`nvidia_caffe.host.libpath`](/frameworks/caffe?id=nvidia_caffehostlibpath- "Basically, it's a LD_LIBRARY_PATH for NVIDIA Caffe in case of a bare metal run.")
[`nvidia_caffe.docker.image`](/frameworks/caffe?id=nvidia_caffedockerimage-hpenvidia_caffecuda9-cudnn7 "The name of a docker image to use for NVIDIA Caffe")

INTEL Caffe
[`intel_caffe.host.path`](/frameworks/caffe?id=intel_caffehostpath- "Path to an Intel Caffe executable in case of a bare metal run.")
[`intel_caffe.host.libpath`](/frameworks/caffe?id=intel_caffehostlibpath-homeprojectsintel_caffebuildtools "Basically, it's a LD_LIBRARY_PATH for Intel Caffe in case of a bare metal run.")
[`intel_caffe.docker.image`](/frameworks/caffe?id=intel_caffedockerimage-hpeintel_caffecpu "The name of a docker image to use for Intel Caffe.")

### __Caffe2__
[`caffe2.docker.image`](/frameworks/caffe2?id=caffe2dockerimage-hpecaffe2cuda9-cudnn7 "The name of a docker image for Caffe2 if containerized benchmark is requested.")
[`caffe2.host.python_path`](/frameworks/caffe2?id=caffe2hostpython_path-homeprojectscaffe2build "Path to a Caffe2's python folder in case of a bare metal run.")
[`caffe2.host.libpath`](/frameworks/caffe2?id=caffe2hostlibpath-homeprojectscaffe2buildcaffe2 "Basically, it's a LD_LIBRARY_PATH for MXNet in case of a bare metal run.")

### __MXNet__
[`mxnet.kv_store`](/frameworks/mxnet?id=mxnetkv_store-device "A method to aggregate gradients")
[`mxnet.docker.image`](/frameworks/mxnet?id=mxnetdockerimage-hpemxnetcuda9-cudnn7 "The name of a docker image to use for MXNet if containerized benchmark is requested.")
[`mxnet.host.python_path`](/frameworks/mxnet?id=mxnethostpython_path-homeprojectsmxnetpython "Path to a MXNET's python folder in case of a bare metal run.")
[`mxnet.host.libpath`](/frameworks/mxnet?id=mxnethostlibpath-quotquot "Basically, it's a LD_LIBRARY_PATH for MXNet in case of a bare metal run.")

### __TensorFlow__
[`tensorflow.var_update`](/frameworks/tensorflow?id=tensorflowvar_update-replicated "This is a 'variable_update' parameter for tf_cnn_benchmarks.")
[`tensorflow.use_nccl`](/frameworks/tensorflow?id=tensorflowuse_nccl-true "This is a 'use_nccl' parameter for tf_cnn_benchmarks.")
[`tensorflow.local_parameter_device`](/frameworks/tensorflow?id=tensorflowlocal_parameter_device-cpu "This is a 'local_parameter_device' parameter for tf_cnn_benchmarks.")
[`tensorflow.docker.image`](/frameworks/tensorflow?id=tensorflowdockerimage-hpetensorflowcuda9-cudnn7 "The name of a docker image to use for TensorFlow if containerized benchmark is requested.")
[`tensorflow.host.libpath`](/frameworks/tensorflow?id=tensorflowhostlibpath-quotquot "Basically, it's a LD_LIBRARY_PATH for TensorFlow in case of a bare metal run.")

### __TensorRT__
[`tensorrt.docker.image`](/frameworks/tensorrt?id=tensorrtdockerimage-hpetensorrtcuda8-cudnn6 "The name of a docker image to use for TensorRT.")
[`tensorrt.host.path`](/frameworks/tensorrt?id=tensorrthostpath-dlbs_rootsrctensorrtbuild "Path to a tensorrt executable in case of bare metal run.")


## Output Parameters
DLBS writes benchmark results along with all input parameters into output log files.
[Log parser](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/dlbs/logparser.py)
with examples [here](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/tutorials/dlcookbook/log_parser.sh)
can be used to parse those files.

By convention, inference/training times are written into `results` name space. This
means all result parameters start with `results.`. For instance, average inference time
for inference benchmarks will be under `results.inference_time` key and training time
for training benchmarks will be under `results.training_time` key.

> We are actively working to provide better experience with output parameters.
  We will introduce this sometime in December 2017 time frame.
