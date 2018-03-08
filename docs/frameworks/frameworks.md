# __Frameworks__

Deep Learning Benchmarking Suite benchmarks frameworks via one or two intermediate scripts:

1. A shell wrapper script (`launcher`) that takes experimenter parameters and calls framework script in host or docker environment. It knows how to translate input parameters into framework specific parameters. Currently, these wrapper scripts are approximately 50-100 lines
2. A script, usually a python project, that knows how to run a framework with specified models. May not be required, like in Caffe case - launcher may directly invoke `caffe` framework. For other frameworks, we usually need a python project that for this. If possible, we take advantage of existing projects (TensorFlow case). In general, it's possible to use these frameworks without DLBS.


## Commonly used configuration parameters
#### __exp.docker__

* __default value__ `True`
* __description__ If true, use docker container to run benchmark. See 'exp.docker_image' for more details..

#### __exp.framework__

* __default value__ `""`
* __description__ Framework to benchmark. Supported frameworks: 'tensorflow', 'caffe2', 'mxnet', 'tensorrt', 'nvidia_caffe', 'intel_caffe', 'bvlc_caffe'.

#### __exp.gpus__

* __default value__ `"0"`
* __description__ A list of GPUs to use. If empty, CPUs should be used instead. Replicas are separated by a ',' character while GPUs within single replica are separated with ':' character, for instance for single node benchmark:   O         No distributed training. Use one model replica on GPU 0   0,1,2,3   Use distributed training with 4 model replicas each occupying one GPU   0:1,2:3   Use distributed training with 2 model replicas. Replica 1 is on GPUs 0 and 1, replica 2 is on GPU 2 and 3. This placement must             be supported by a benchmarking script and specific model. In case of multi-node training, this parameter defines a model placement on one node assuming each node uses the same model to GPU placement.

#### __exp.log_file__

* __default value__ `"$('${exp.gpus}'.replace(',', '.'))$_${exp.model}_${exp.effective_batch}.log"`
* __description__ The name of a log file for this experiment.

#### __exp.model__

* __default value__ `""`
* __description__ A neural network model to benchmark. Valid values include 'alexnet', 'googlenet', 'resnet50' etc. In general, not all frameworks can support all models. Refer to documentation \(section 'models'\) on what frameworks support what models.

#### __exp.num_batches__

* __default value__ `100`
* __description__ Number of benchmark batches to perform. Based on average batch time, experimenter will compute performance.

#### __exp.num_warmup_batches__

* __default value__ `1`
* __description__ Number of warmup batches to process before starting measuring performance. May not be supported by all frameworks.

#### __exp.phase__

* __default value__ `"training"`
* __description__ Phase to benchmark. Possible values - 'inference' or 'training'.

#### __exp.replica_batch__

* __default value__ `16`
* __description__ A replica batch size. This is something that's called a device batch size. Assuming we will in future be able to benchmark models that do not fit into one GPU and single replica will require multiple GPUs, a device batch does not clearly represent situation in this case.


## Other parameters
#### __exp.cuda__

* __default value__ `""`
* __description__ In case of NVIDIA GPUs, the version of CUDA.

#### __exp.cudnn__

* __default value__ `""`
* __description__ In case of NVIDIA GPUs, the version of cuDNN.

#### __exp.data__

* __default value__ `"synthetic"`
* __description__ Indicator if real or synthetic data was used in experiment. Real data means the presence of input injection pipeline synthetic data means no injection pipeline. Input tensors are initialized with random numbers. The data specificatio is usually a framework specific and typically specified by a 'data_dir' parameter in a respective framework namespace i.e. 'tensorflow.data_dir'.

#### __exp.data_store__

* __default value__ `""`
* __description__ An identifier of a data location. May include such values as 'mem' \(in memory\), 'local-hdd' \(data is located on a locally attached  hdd\), 'local-ssd', 'remote-hdd' etc. This value needs to be provided by a user. Only useful if real data was used \(see 'exp.data'\)

#### __exp.device_title__

* __default value__ `""`
* __description__ A title \(name, model\) of a main compute device. Something like 'P100 PCIe', 'V100 NVLINK', 'P4', 'E5-2650 v2' etc.

#### __exp.device_type__

* __default value__ `"$('gpu' if ${exp.num_gpus} > 0 else 'cpu')$"`
* __description__ Type of main compute device - 'gpu' or 'cpu'.

#### __exp.docker_args__

* __default value__ `"--rm"`
* __description__ Additional arguments to pass to docker. For instance, you may want to pass these NVIDIA reccommended parameters:   --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864

#### __exp.docker_image__

* __default value__ `"${${exp.framework}.docker_image}"`
* __description__ Docker image to use for current benchmark. Must exist in the system.

#### __exp.docker_launcher__

* __default value__ `"$('nvidia-docker' if '${exp.device_type}' == 'gpu' else 'docker')$"`
* __description__ A launcher for docker containers - 'nvidia-docker' for GPU workloads and 'docker' for CPU ones.

#### __exp.dtype__

* __default value__ `"float32"`
* __description__ Type of data to use if supported by a framework. Possible values 'float', 'float32', 'float16' or 'int8'. The 'float' and 'float32' means the same - 32 bit floating point numbers.

#### __exp.effective_batch__

* __default value__ `"$(${exp.num_replicas}*${exp.replica_batch} if '${exp.device_type}' == 'gpu' else ${exp.num_nodes} * ${exp.replica_batch})$"`
* __description__ Effective batch size. By default, it is computed based on 'exp.replica_batch' what makes weak scaling exploration a default choice.

#### __exp.framework_commit__

* __default value__ `""`
* __description__ A head commit of a framework that's used in benchmarks. Can be useful for reproducible experiments. The support for automatic identification of the head commit is limited now.

#### __exp.framework_family__

* __default value__ `"${exp.framework}"`
* __description__ A framework family identifier. In most situations this is the same as a framework identifier \('exp.framework'\). For Caffe's forks \(bvlc_caffe, nvidia_caffe and intel_caffe\) it is 'caffe'. This parameter is used to identify what launcher is responsible for running benchmarks. All Caffe's forks share the same launcher.

#### __exp.framework_title__

* __default value__ `""`
* __description__ Human readable framework title that goes into reports. Something like 'TensorFlow', 'Caffe2', 'MXNet', 'TensorRT', 'NVIDIA Caffe', 'Intel Caffe' and 'BVLC Caffe'

#### __exp.framework_ver__

* __default value__ `""`
* __description__ Framework version. Usually, the value is automatically identified based on actual framework used in benchmarks \(either bare metal or docker\).

#### __exp.id__

* __default value__ `"$(uuid.uuid4().__str__().replace('-', ''))$"`
* __description__ Unique identifier \(UUID\) of an experiment. Can be used to uniqely identify individual benchmarks.

#### __exp.model_title__

* __default value__ `""`
* __description__ A human readable neural network model that goes into reports. Possible values are 'AlexNet', 'GoogleNet', 'ResNet50' etc.

#### __exp.node_id__

* __default value__ `""`
* __description__ The name of a node. Should be used to identify various servers. Something like 'apollo6500', 'DL380', 'DGX1' etc.

#### __exp.node_nic__

* __default value__ `""`
* __description__ If distributed benchmark, a string description of an interconnect. Something like 'EDR', 'FDR', '1GB Ethernet' etc.

#### __exp.num_gpus__

* __default value__ `"$(${exp.num_local_gpus} * ${exp.num_nodes})$"`
* __description__ Total number of all GPUs on all nodes in distributed training.

#### __exp.num_local_gpus__

* __default value__ `"$(len(re.sub('[:,]', ' ', '${exp.gpus}').split()))$"`
* __description__ Total number of GPUs to use on one node.

#### __exp.num_local_replicas__

* __default value__ `"$(len('${exp.gpus}'.replace(',', ' ').split()) if '${exp.device_type}' == 'gpu' else 1)$"`
* __description__ Number of model replicas in distributed benchmark on one node.

#### __exp.num_nodes__

* __default value__ `1`
* __description__ Number of nodes in case of multi-node benchmark.

#### __exp.num_replicas__

* __default value__ `"$(${exp.num_local_replicas} * ${exp.num_nodes})$"`
* __description__ Total number of replicas on all nodes in distributed benchmark.

#### __exp.proj__

* __default value__ `""`
* __description__ An optional project identifier. Can be used to logically group invidual benchmarks. It enables clean and easy way to retrieve result from log files / database. To select all benchmarks logically grouped under 'my_project_identifier' project, one may use something like:   select ... from ... where exp.proj = 'my_project_identifier'. To identify individial benchmarks within series of benchmarks in one project, use 'exp.id' parameter.

#### __exp.rerun__

* __default value__ `False`
* __description__ By default, if experimenter finds existing file for an experiment \(see 'exp.log_file'\), it will not run experiment again. Set the value of this parameter to 'true' to force rerun benchmarks in this case.

#### __exp.status__

* __default value__ `"ok"`
* __description__ If 'disabled' on input, experimenter will not run this benchmark. In other cases, it will contain status code. The parameter 'exp.status_msg' may contain additional textual description. Possible values:   On input:     disabled   Experiment will not run \(probably, disabled by an extension\).     simulate   Print command line arguments and do not run. Useful to debug computations of parameters.   On output:     ok         Experiment has been completed successfully.     skipped    Experiment has not been conducted. Possible reasons - log file exists \(and not force to rerun\) or batch size is too large                \(this may be known based on previous benchmarks for similar configurations\).                UPDATE: This code is not set if experiment has already been done \(log file exists\). In this case whatever code is in this                        log file is returned.     failure    Experiment has failed. See parameter 'exp.status_msg' that may contain additional details.

#### __exp.status_msg__

* __default value__ `""`
* __description__ A textual description for a status code stored in parameter 'exp.status'. This may be empty if certain log parsers have not been implemented yet. In future releases, this may become an object that will contain textual description and advanced information such as thrown exceptions.

#### __exp.sys_info__

* __default value__ `""`
* __description__ A comma separated string that defines what tools should be used to collect system wide information. A default empty value means no system information is collected. To collect all information use:     -Pexp.sys_info='"inxi,cpuinfo,meminfo,lscpu,nvidiasmi"' The following source of information are supported:     inxi       The inxi must be available \(https://github.com/smxi/inxi\). It is an output of 'inxi -Fbfrlp'.     cpuinfo    Content of /proc/cpuinfo     meminfo    Content of /proc/meminfo     lscpu      Output of 'lscpu'     nvidiasmi  Output of '/usr/bin/nvidia-smi -q' The information is stored in a 'hw' namespace i.e. hw.inxi, hw.cpuinfo, hw.meminfo, hw.lscpu and hw.nvidiasmi. In addition, a complete output in a json format can be obtained with:     python ./python/dlbs/experimenter.py sysinfo

#### __exp.use_tensor_core__

* __default value__ `True`
* __description__ If true, enable tensor core operations for NVIDIA V100 and CUDA >= 9.0 if supported by a framework.

#### __monitor.backend_pid_folder__

* __default value__ `"$('${monitor.pid_folder}' if not ${exp.docker} else '/workspace/tmp')$"`
* __description__ This is a host or docker folder that will be used by a benchmarking scripts. Will be different from `monitor.pid_folder` if containerized benchmark is performed. Users must not change this parameter.

#### __monitor.frequency__

* __default value__ `0`
* __description__ A sampling frequency in seconds of embedded resource monitor. By default \(0\) resource monitor is disabled. If this value is > 0, experimenter will start embedded resource monitor \(kind of a reference implementation\) that will log system parameters with this frequency. This parameters include cpu and memory consumption, power, GPU metrics etc. Assumption: in current implementation, if resource monitor is enabled for a first benchmark, it's considered to be enabled for rest of benchmarks and vice versa.

#### __monitor.launcher__

* __default value__ `"${DLBS_ROOT}/scripts/resource_monitor.sh"`
* __description__ A path to an embedded resource monitor.

#### __monitor.pid_folder__

* __default value__ `"/dev/shm/monitor"`
* __description__ A host folder that will be used by a resource monitor and benchmarking scripts to communicate process id that should be monitored. Users need to specify this parameter of they want to change default path.

#### __monitor.timeseries__

* __default value__ `"time:str:1,mem_virt:float:2,mem_res:float:3,mem_shrd:float:4,cpu:float:5,mem:float:6,power:float:7,gpus:float:8:"`
* __description__ A string that specifies which timeseries metrics must go into a log file. Metrics are separated with comma \(,\). Each metric specification consists of three or four fields separated with colon \(:\) - 'name:type:index_range'. The name specifies timeseries name. The field in log file will be composed as 'results.use.$name'. Type specifies how values that come from monitor need to be cast \(std, int, float or bool\). Values from resource monitor come as a whitespace separated string. The index range specifies how that maps to a timeseries name. It can be a single integer\(for instance time:str:1\) specfying exact index or a index and number of elements that should be appended to a timeseries item. Number of elements may not be present what means scan until the end of list is reached  \(for instance gpu:float:8:2 or gpu:float:8:\). If number of elements is specified, a timeseries will contain items that will be lists event though number of elements may be 1.

#### __runtime.EXPORT_CUDA_CACHE_PATH__

* __default value__ `"CUDA_CACHE_PATH=$(('${runtime.cuda_cache}' if not ${exp.docker} else '/workspace/cuda_cache') if '${runtime.cuda_cache}' else '')$"`
* __description__ A variable that can be used in framework configurations to set CUDA cache path. Not used by default.

#### __runtime.EXPORT_CUDA_VISIBLE_DEVICES__

* __default value__ `"CUDA_VISIBLE_DEVICES=${runtime.visible_gpus}"`
* __description__ A variable that can be used in framework configurations to make this GPUs visible. Not used by default.

#### __runtime.cuda_cache__

* __default value__ `"/dev/shm/dlbs"`
* __description__ If not empty, use this folder as a CUDA cache \(search for CUDA_CACHE_PATH environmental variable\).

#### __runtime.launcher__

* __default value__ `""`
* __description__ A sequence of commands that need to be placed on a command line right before benchmarking scripts. Can be used to pin process to certain CPUs \(numactl, taskset\).

#### __runtime.visible_gpus__

* __default value__ `"$(re.sub('[:]', ',', '${exp.gpus}'))$"`
* __description__ A list of GPUs that should be made visible to benchmarking scripts.




## Framework specific configuration parameters

Navigate to framework section to learn more about framework specific parameters.
Also, these frameworks section provide more details on backend implementation and
integration with DLBS.

* [Caffe](/frameworks/caffe.md?id=caffe)
* [Caffe2](/frameworks/caffe2.md?id=caffe2)
* [MXNet](/frameworks/mxnet.md?id=mxnet)
* [TensorFlow](/frameworks/tensorflow.md?id=tensorflow)
* [TensorRT](/frameworks/tensorrt.md?id=tensorrt)
* [PyTorch](/frameworks/pytorch.md?id=pytorch)
