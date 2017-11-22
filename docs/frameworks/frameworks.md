# __Frameworks__

Deep Learning Benchmarking Suite benchmarks frameworks via one or two intermediate scripts:

1. A shell wrapper script (`launcher`) that takes experimenter parameters and calls framework script in host or docker environment. It knows how to translate input parameters into framework specific parameters. Currently, these wrapper scripts are approximately 50-100 lines
2. A script, usually a python project, that knows how to run a framework with specified models. May not be required, like in Caffe case - launcher may directly invoke `caffe` framework. For other frameworks, we usually need a python project that for this. If possible, we take advantage of existing projects (TensorFlow case). In general, it's possible to use these frameworks without DLBS.

## Commonly used configuration parameters
These parameters can be used with all frameworks and are commonly used. In 95% of
benchmarks only these parameters need to be used (additionally, see frameworks
sections for framework specific parameters).

### __exp.framework__ = `""`
Framework to use. Supported frameworks: `tensorflow`, `caffe2`, `mxnet`, `tensorrt`,
`nvidia_caffe`, `intel_caffe`, `bvlc_caffe`. May be overridden be experimenter if multiple",
"frameworks share the same backend implementation like all Caffe forks.
### __exp.model__ = `""`
A model identifier to use such as `alexnet`, `googlenet`, `vgg16` etc. Framework specific -
not all frameworks may support all models.
### __exp.env__ = `host`
Benchmarking environment - docker or bare metal. Possible values: `docker` or `host`.
### __exp.warmup_iters__ = `2`
Number of warmup iterations to perform if supported by backend (For instance, Caffe does not
support it when running in inference phase.
### __exp.bench_iters__ = `100`
Number of benchmarking iterations. Average time is reported based on this number of
iterations.
### __exp.phase__ = `training`
Phase to benchmark. Possible values - `inference` or `training`.
### __exp.device_batch__ = `16`
A device batch size. Effective batch size is computed as this number multiplied by
a total number of compute devices (for instance, GPUs).
### __exp.gpus__ = `0`
GPUs to use. List of comma-separated GPU identifiers. For instance `0` or `0,1`
or `0,1,2,3` or `0,1,2,3,4,5,6,7`.
### __exp.log_file__ = `${exp.exp_path}/exp/$('${exp.gpus}'.replace(',','.'))$_${exp.model}_${exp.effective_batch}.log`
Benchmark log file.

## Other parameters
Less frequently used parameters or parameters that are set automatically for
internal use.

### __exp.dtype__ = `float`
Type of data to use if supported by a framework. Possible values `float32`(`float`),
`float16` or `int8`.
### __exp.num_gpus__ = `$(len('${exp.gpus}'.replace(',', ' ').split()))$`
Number of GPUs. Default value is computed based on _exp.gpus_ value.
### __exp.device__ = `$('gpu' if ${exp.num_gpus} > 0 else 'cpu')$`
Device to use. Possible values - `gpu` or `cpu`. By default, is computed based on
_exp.num_gpus_ value.
### __exp.enable_tensor_core__ = `false`
If `true`, enable tensor core operations for NVIDIA V100 and CUDA >= 9.0 if supported
by a framework. Possible values `true` or `false`.
### __exp.simulation__ = `false`
If `true`, do not run benchmark but print framework command line to a log file instead.
### __exp.bench_root__ = `${BENCH_ROOT}`
Root benchmark folder. Based on this path other paths, like log files, may be specified.
The _BENCH_ROOT_ is an environmental variable that may be used.
### __exp.framework_id__ = `${exp.framework}`
Unique framework identifier, default value is _exp.framework_. In most situations,
they are the same. However, for Caffe's forks they are different. The _exp.framework_
initially, for instance, may equal to `bvlc_caffe`. After initialization, the
_exp.framework_id_ will be equal to `bvlc_caffe` and _exp.framework_ will become `caffe`.
### __exp.id__ = `$(uuid.uuid4().__str__().replace('-',''))$`
UUID for a single benchmark experiment.
### __exp.effective_batch__ = `$(${exp.num_gpus}*${exp.device_batch} if '${exp.device}' == 'gpu' else ${exp.device_batch})$`
Effective batch size.
### __exp.exp_path__ = `${exp.bench_root}/${exp.framework}/${exp.env}/${exp.device}/${exp.phase}`
Root folder where benchmark log files are stored. It's not required to use this parameter.
The file name that is used to log benchmarks is _exp.log_file_.
### __exp.force_rerun__ = `false`
If benchmark log file exists and this value is `false`, benchmark will not be ran.
### __exp.docker.launcher__ = `$('nvidia-docker' if '${exp.device}' == 'gpu' else 'docker')$`
One of `nvidia-docker` or `docker` depending on a _exp.device_ value.
### __resource_monitor.enabled__ = `false`
If `true`, in-process monitor is started that logs system resource consumption. This
includes CPU and memory utilization, GPU power consumption and utilization, server power
consumption tracking if supported by a server.
### __resource_monitor.pid_file_folder__ = `/dev/shm/dl`
A folder that contains file that is used to communicate process identifier to monitor.
### __resource_monitor.launcher__ = `${DLBS_ROOT}/scripts/resource_monitor.sh`
A resource monitor launcher script.
### __resource_monitor.data_file__ = `${exp.bench_root}/resource_consumption.csv`
A resource monitor log file. This file will contain time series with measurements.
### __resource_monitor.frequency__ = `0.1`
Sampling frequency in seconds. May be equal to something like `0.1`.
### __runtime.limit_resources__ = ``
Something that limits process resources. It's used like '${runtime.limit_resources} ${runtime.bind_proc}
command with parameters'.
### __runtime.bind_proc__ = ``
Can be used to specify process binding commands like 'numactl' or 'taskset'.
In general, it's any command that should launch the framework. May be a debugger probably.
It's used like '${runtime.limit_resources} ${runtime.bind_proc} command with parameters'."
### __runtime.cuda_cache__ = `$('${CUDA_CACHE_PATH}' if '${exp.env}' == 'host' else '/workspace/cuda_cache')$`
CUDA cache path. May significantly speedup slow startup when a large number of
experiments are ran. Set it to somewhere in '/dev/shm'. Default value is based on
environmental variable CUDA_CACHE_PATH.
### __sys.plan_builder.var_order__ = `["exp.framework", "exp.phase", "exp.model", "exp.gpus"]`
Order in which plan builder varies variables doing Cartesian product.
### __sys.plan_builder.method__ = `cartesian_product`
Method to build multiple experiments, the only supported value is `cartesian_product`.

## Framework specific configuration parameters

Navigate to framework section to learn more about framework specific parameters.
Also, these frameworks section provide more details on backend implementation and
integration with DLBS.

* [Caffe](/frameworks/caffe.md?id=caffe)
* [Caffe2](/frameworks/caffe2.md?id=caffe2)
* [MXNet](/frameworks/mxnet.md?id=mxnet)
* [TensorFlow](/frameworks/tensorflow.md?id=tensorflow)
* [TensorRT](/frameworks/tensorrt.md?id=tensorrt)
