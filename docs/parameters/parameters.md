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
   distributed aggregation schema etc.
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
python experimenter.py help --params exp.device_type
python experimenter.py help --params exp.*

# Perform full-text search in parameters description, case insensitive
python experimenter.py help --text cuda

# Perform full-text search in a subset of parameters that match params
python experimenter.py help --params batch --text replica

# Show most commonly used parameters for TensorFlow
experimenter.py help --frameworks tensorflow
```

## Commonly used parameters

### __Common parameters for all frameworks__
[`exp.framework`](/frameworks/frameworks?id=expframework "Framework to benchmark. Supported frameworks: 'tensorflow', 'caffe2', 'mxnet', 'tensorrt', 'nvidia_caffe', 'intel_caffe', 'bvlc_caffe'.")
[`exp.model`](/frameworks/frameworks?id=expmodel "A neural network model to benchmark. Valid values include 'alexnet', 'googlenet', 'resnet50' etc. In general, not all frameworks can support all models. Refer to documentation \(section 'models'\) on what frameworks support what models.")
[`exp.docker`](/frameworks/frameworks?id=expdocker "If true, use docker container to run benchmark. See 'exp.docker_image' for more details..")
[`exp.num_warmup_batches`](/frameworks/frameworks?id=expnum_warmup_batches "Number of warmup batches to process before starting measuring performance. May not be supported by all frameworks.")
[`exp.num_batches`](/frameworks/frameworks?id=expnum_batches "Number of benchmark batches to perform. Based on average batch time, experimenter will compute performance.")
[`exp.phase`](/frameworks/frameworks?id=expphase "Phase to benchmark. Possible values - 'inference' or 'training'.")
[`exp.replica_batch`](/frameworks/frameworks?id=expreplica_batch "A replica batch size. This is something that's called a device batch size. Assuming we will in future be able to benchmark models that do not fit into one GPU and single replica will require multiple GPUs, a device batch does not clearly represent situation in this case.")
[`exp.gpus`](/frameworks/frameworks?id=expgpus "A list of GPUs to use. If empty, CPUs should be used instead. Replicas are separated by a ',' character while GPUs within single replica are separated with ':' character, for instance for single node benchmark:   O         No distributed training. Use one model replica on GPU 0   0,1,2,3   Use distributed training with 4 model replicas each occupying one GPU   0:1,2:3   Use distributed training with 2 model replicas. Replica 1 is on GPUs 0 and 1, replica 2 is on GPU 2 and 3. This placement must             be supported by a benchmarking script and specific model. In case of multi-node training, this parameter defines a model placement on one node assuming each node uses the same model to GPU placement.")
[`exp.log_file`](/frameworks/frameworks?id=explog_file "The name of a log file for this experiment.")

### __TensorFlow__
[`tensorflow.var_update`](/frameworks/tensorflow?id=tensorflowvar_update "This is a 'variable_update' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.")
[`tensorflow.use_nccl`](/frameworks/tensorflow?id=tensorflowuse_nccl "This is a 'use_nccl' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.")
[`tensorflow.local_parameter_device`](/frameworks/tensorflow?id=tensorflowlocal_parameter_device "This is a 'local_parameter_device' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.")
[`tensorflow.docker_image`](/frameworks/tensorflow?id=tensorflowdocker_image "The name of a docker image to use for TensorFlow if containerized benchmark is requested.")
[`tensorflow.host_libpath`](/frameworks/tensorflow?id=tensorflowhost_libpath "Basically, it's a LD_LIBRARY_PATH for TensorFlow in case of a bare metal run.")

### __BVLC Caffe__
[`bvlc_caffe.host_path`](/frameworks/caffe?id=bvlc_caffehost_path "Path to a BVLC Caffe executable in case of a bare metal run.")
[`bvlc_caffe.host_libpath`](/frameworks/caffe?id=bvlc_caffehost_libpath "Basically, it's a LD_LIBRARY_PATH for BVLC Caffe in case of a bare metal run.")
[`bvlc_caffe.docker_image`](/frameworks/caffe?id=bvlc_caffedocker_image "The name of a docker image to use for BVLC Caffe.")

### __NVIDIA Caffe__
[`nvidia_caffe.host_path`](/frameworks/caffe?id=nvidia_caffehost_path "Path to a NVIDIA Caffe executable in case of a bare metal run.")
[`nvidia_caffe.host_libpath`](/frameworks/caffe?id=nvidia_caffehost_libpath "Basically, it's a LD_LIBRARY_PATH for NVIDIA Caffe in case of a bare metal run.")
[`nvidia_caffe.docker_image`](/frameworks/caffe?id=nvidia_caffedocker_image "The name of a docker image to use for NVIDIA Caffe.")

### __Intel Caffe__
[`intel_caffe.host_path`](/frameworks/caffe?id=intel_caffehost_path "Path to an Intel Caffe executable in case of a bare metal run.")
[`intel_caffe.host_libpath`](/frameworks/caffe?id=intel_caffehost_libpath "Basically, it's a LD_LIBRARY_PATH for Intel Caffe in case of a bare metal run.")
[`intel_caffe.docker_image`](/frameworks/caffe?id=intel_caffedocker_image "The name of a docker image to use for Intel Caffe.")

### __Caffe2__
[`caffe2.docker_image`](/frameworks/caffe2?id=caffe2docker_image "The name of a docker image for Caffe2 if containerized benchmark is requested.")
[`caffe2.host_python_path`](/frameworks/caffe2?id=caffe2host_python_path "Path to a Caffe2's python folder in case of a bare metal run.")
[`caffe2.host_libpath`](/frameworks/caffe2?id=caffe2host_libpath "Basically, it's a LD_LIBRARY_PATH for Caffe2 in case of a bare metal run.")

### __MxNet__
[`mxnet.kv_store`](/frameworks/mxnet?id=mxnetkv_store "A method to aggregate gradients \(local, device, dist_sync, dist_device_sync, dist_async\). See https://mxnet.incubator.apache.org/how_to/multi_devices.html for more details.")
[`mxnet.docker_image`](/frameworks/mxnet?id=mxnetdocker_image "The name of a docker image to use for MXNet if containerized benchmark is requested.")
[`mxnet.host_python_path`](/frameworks/mxnet?id=mxnethost_python_path "Path to a MXNET's python folder in case of a bare metal run.")
[`mxnet.host_libpath`](/frameworks/mxnet?id=mxnethost_libpath "Basically, it's a LD_LIBRARY_PATH for MXNet in case of a bare metal run.")

### __PyTorch__
[`pytorch.docker_image`](/frameworks/pytorch?id=pytorchdocker_image "The name of a docker image to use for PyTorch if containerized benchmark is requested.")
[`pytorch.cudnn_benchmark`](/frameworks/pytorch?id=pytorchcudnn_benchmark "Uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. If this is set to false, uses some in-built heuristics that might not always be fastest. By default cudnn_benchmark is set to TRUE. Setting to true will improve performance, at the expense of using more memory. The input shape should be the same for each batch, otherwise autotune will re-run for each batch, causing a huge slow-down. More details are here: https://github.com/soumith/cudnn.torch#modes")
[`pytorch.cudnn_fastest`](/frameworks/pytorch?id=pytorchcudnn_fastest "Enables a fast mode for the Convolution modules - simply picks the fastest convolution algorithm, rather than tuning for workspace size. By default, cudnn.fastest is set to false. You should set to true if memory is not an issue, and you want the fastest performance. More details are here: https://github.com/soumith/cudnn.torch#modes")

### __TensorRT__
[`tensorrt.docker_image`](/frameworks/tensorrt?id=tensorrtdocker_image "The name of a docker image to use for TensorRT.")
[`tensorrt.host_path`](/frameworks/tensorrt?id=tensorrthost_path "Path to a tensorrt executable in case of bare metal run.")



## Output Parameters
DLBS writes benchmark results along with all input parameters into output log files.
[Log parser](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/dlbs/logparser.py)
with examples [here](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/tutorials/dlcookbook/log_parser.sh)
can be used to parse those files.

By convention, inference/training times are written into `results` name space. This
means all result parameters start with `results.`. For instance, average training/inference time will be under `results.time` key (the `exp.phase` parameter will specify if it's inference/training phase).

> We are actively working to provide better experience with output parameters.
  We will introduce this sometime in December 2017 time frame.
