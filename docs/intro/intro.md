# __Introduction__

Every benchmark is defined by a set of parameters. Every parameter is a key-value pair. Parameters define everything - frameworks, batch sizes, log files, models etc. Benchmark can be defined manually be enumerating all parameters, or automatically, by providing a configuration that defines how parameters should vary from one experiment to another. A script called [exerimenter](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/dlbs/experimenter.py) varies parameters (basically, by doing Cartesian product) and creates multiple benchmarks with different parameters.

<p align="center">
  <img src="https://raw.githubusercontent.com/HewlettPackard/dlcookbook-dlbs/master/docs/intro/imgs/overview.png">
</p>

Every framework that we support has a wrapper script - a short 40-80 line bash file that knows how to run benchmarks for a specific framework (Caffe's forks do not have it). Experimenter launches this script and passes all benchmark parameters on a command line. Bash wrapper script then parses these parameters and launches a framework.

## Architecture overview
There are several components in DLBS:
1. `Experimenter` - a single entry point python script that's responsible of generating benchmark configurations and launching benchmark scripts. Input parameters to this script are benchmark specifications in a json format. Alternatively, benchmark specifications can be passed on a command line. This script is responsible of generating all benchmark configurations and launching benchmarks one at a time. It uses a special wrapper scripts (currently, written in bash) for that purpose. It launches script and passes to that script all parameters of a current experiment.

2. `Launcher script` - aka launcher - a thin bash wrapper script that launches framework with appropriate arguments. Its task is to parse command line arguments and launch respective framework using these parameters. If supported, this script must be able to launch experiments in docker and host environments or in distributed environment.

3. `Benchmark script` - if present, usually a python project that knows how to run training/inference with a particular model for one framework. We take advantage of existing TensorFlow's CNN [benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) to benchmark TensorFlow. We have our in-house implementation of benchmarking projects with similar command line API for [Caffe2](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/caffe2_benchmarks), and [MXNet](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/mxnet_benchmarks) frameworks. Caffe's forks do not require similar benchmark scripts due to natural integration with launcher scripts.

4. `Log parser`. Every entity in DLBS writes information to log files as key-value pairs. Log parser parses any textual file and extracts those key-value pairs and puts them into a dictionary. A value can be an arbitrary json parsable string (i.e. values can be any json-serialziable objects like numbers, strings, lists, dictionaries, objects etc.)

5. `Report builder`. A tool that can generate simple reports outlining weak/strong scaling results of performed experiments.


## Benchmark specification
Every benchmark is specified by a set of parameters. Some of those parameters have specific meaning. Otherwise, arbitrary parameters can be used. For instance, the following json object defines a benchmark to run GoogleNet in TensorFlow with batch equal to 8 images per model replica (device if no model parallelism is used):
```json
{
    'exp.model': 'googlenet',
    'exp.framework': 'tensorflow',
    'exp.replica_batch': 8
}
```
It is assumed that for every framework (`exp.framework`) there must be a parameter with name `framework-name.launcher`, for instance, `tensorflow.launcher`, `caffe.launcher`, `caffe2.launcher` etc. Experimenter will run this script and will pass all parameters from current experiment as command line arguments. Since we cannot use '.' in parameter names, all dots will be converted to underscores '\_'. For instance, if TensorFlow is configured with the following launcher:

```json
{ 'tensorflow.launcher': 'tensorflow_hpm.sh' }
```
then the experimenter will launch the following process:

```bash
tensorflow_hpm.sh --exp_model 'googlenet'  --exp_framework 'tensorflow' --exp_replica_batch 8
```
The script `tensorflow_hpm.sh` can then parse command line arguments and access these parameters. In general, there can be many parameters passed to a particular launcher. You should only use those that are required to run benchmark in a specific framework.


## Benchmark generation
Instead of providing manually configurations for every benchmark, experimenter can generate those configurations automatically based on input specifications. The format of input specification in json format is the following:
```json
{
    'parameters':{},
    'variables': {},
    'extensions':[
        {
            'condition':{},
            'parameters': {},
            'cases': [
                {}
            ]
        }
    ]
}
```
### Parameters
`Parameters` section defines parameters that do not need to be varied in different experiments.
Typical examples for such parameters are number of warm-up and benchmark iterations or docker images for various frameworks. If some of these parameters need to be varied, use `variables` section to define these parameters. This is an example configuration:
```json
{
    'parametes':{
        'exp.num_warmup_batches': 2,
        'exp.num_batches': 100,
        'exp.phase': 'training'
    }
}
```

### Variables
`Variables` section defines parameters that need to be varied in experiment. For instance, frameworks, models and batch size can go here. If for some reason you do not need to vary them, have just one value for them or move them to `parameters` section.
```json
{
    'variables':{
        'exp.framework': ['tensorflow', 'caffe2', 'tensorrt'],
        'exp.replica_batch': [8, 16, 32, 64],
        'exp.phase': ['training', 'inference']
    }
}
```
Experimenter script will take all these parameters from `variables` section and will compute Cartesian product. Thus, in the given example there will totally be `3*4*2=24` benchmark configurations. With every configuration the experimenter will do the following:

1. It will create copy of parameters
2. It will add generated configuration to parameters, possible, overwriting existing ones.

This is the way how we can explore various combinations and how different parameters affect performance. This, however, is not flexible enough. We support so called `extensions` that define parameters or sets of parameters that can override existing parameters or add new benchmark configurations.

### Extensions
Every extension has `condition`, `parameters` and `cases` sections. Let's consider the following example:
```json
    "extensions": [
        {
            "condition":{ "exp.framework": "bvlc_caffe" },
            "parameters": {
                "exp.framework_title":"BVLC Caffe",
                "exp.framework_family": "caffe",
                "caffe.fork": "bvlc"
            }
        },
        {
            "condition":{ "exp.framework": "nvidia_caffe" },
            "parameters": {
                "exp.framework_title":"NVIDIA Caffe",
                "exp.framework_family": "caffe",
                "caffe.fork": "nvidia"
            }
        }
    ]
```
There are two extensions defined in this configuration. First extension is fired when current configuration contains `exp.framework` parameter with value `bvlc_caffe`. In this case, it will take parameters from extension's parameter section and put them into current configuration, possibly, overwriting existing values. The second extension is fired when `exp.parameter` parameter equals to `nvidia_caffe`.

The `cases` section, an array, contains sets of parameters that should be used to extend current configuration. Let's assume that some extension has two cases `A` and `B` each containing several parameters. The experimenter will then copy existing configuration and will add parameters from section `A` to one configuration and parameters from section `B` to another. Thus, extensions allow creating multiple configurations from current configuration.

## Parameter specifications

Parameters are defined as key-value pairs. To some extent, values can reference other parameters, similar to what it is possible in shell scripts. The experimenter will then try to expand these variables. One parameter can reference other parameter with a standard `${}` notation (but not `$name`). It will be resolved to (1) other parameter defined in current experiment and if failed (no parameter with given name), experimenter will try to find this  parameter in a set of environmental variables. If not found, exception is thrown and program terminates. For instance, the  following parameters after evaluation will have the same value:
```json
{
    'exp.framework': 'tensorflow',
    'exp.framework_family': '${exp.framework}'
}
```
We do not support `$name` expansion. It is ignored. One use case for such behavior is to, for instance, generate unique log files:
```json
{"exp.exp_path": "${BENCH_ROOT}/${exp.framework}/${exp.env}/${exp.device}/${exp.phase}"}
```
The second option to define values dynamically is python statements. The format is like this: `$(...)$`. For instance:
```json
{
    "exp.log_file": "${exp.exp_path}/exp/$('${exp.gpus}'.replace(',','.'))$_${exp.model}_${exp.effective_batch}.log",
	"exp.effective_batch": "$(${exp.num_replicas}*${exp.replica_batch} if '${exp.device_type}' == 'gpu' else ${exp.num_nodes} * ${exp.replica_batch})$",
	"exp.num_local_gpus": "$(len(re.sub('[:,]', ' ', '${exp.gpus}').split()))$",
	"exp.device": "$('gpu' if ${exp.num_gpus} > 0 else 'cpu')$",
	"exp.id": "$(uuid.uuid4().__str__().replace('-',''))$"
}
```
If some of the parameters cannot be expanded or if there is a cyclic dependency, experimenter will exit.

> Extensions introduces flexibility in a process of designing benchmarks. However, it can be a source of errors since there are certain rules that need to be followed. Else, you can get results you do not expect. One option to somewhat verify that everything is what you expect is to build a plan and serialize it to file. Then, study the plan and see if benchmark configurations make sense. For more details on plan, see dlcookbook tutorial references at the end of this page.


## Default parameters
We have default parameters for Caffe, Caffe2, TensorFlow, MXNet and TensorRT frameworks. They are located in [configs](https://github.hpe.com/labs/dlcookbook/tree/master/python/dlbs/configs) folder and are loaded each time you run experimenter. So, there's no need to provide every parameter. If required, you can disable loading default parameters and create configurations from scratch.

## Command line arguments
It is not required to provide json specification to experimenter. Majority of benchmark experiments can be specified on a command line.

Parameters are defined with `-P` argument and variables are defined with `-V` argument:
```bash
python $script build -Pexp.num_batches=1000\
                     -Vexp.framework='["tensorflow", "caffe2"]'\
                     -Vexp.replica_batch='[1, 2, 4, 8]'
```
In this example we define one parameter `exp.num_batches` with value `1000` and variable parameters `exp.framework` and `exp.replica_batch`.

Extensions can also be defined on a command line though complex extensions may be quite long and that may lead to errors:
```bash
python $script build   --discard-default-config --log-level=debug\
                       -Vexp.framework='"dummy"'\
                       -Vexp.greeting='["Hello!", "How are you?"]'\
                       -Pdummy.launcher='"${DLBS_ROOT}/scripts/launchers/dummy.sh"'\
                       -E'{"condition":{"exp.greeting":"Hello!"}, "parameters": {"exp.greeting.extension": "You should see me only when exp.greeting is Hello!"}}'
```
In this example we define one condition that will fire when `exp.greeting` is equal to `Hello!`. If so, it will add parameter `exp.greeting.extension` with value `You should see me only when exp.greeting is Hello!` to current set of parameters. Basically, extension in this example is a json parsable string.

> Make sure that parameter/variable values are JSON parseable.

## Try it yourself
The best way to get yourself familiar with this tool is to actually run something. In the [tutorials](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/tutorials/dlcookbook) folder we provide multiple scripts that demonstrate how this tool works.

Every script has detailed comments and multiple examples. Just go through those files, read comments, uncomment lines that actually run experimenter with  various parameters and observe output.

1. [introduction.sh](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/tutorials/dlcookbook/introduction.sh) This is the first script that introduces experimenter and its command line arguments API.
2. [bvlc_caffe.sh](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/tutorials/dlcookbook/bvlc_caffe.sh), [caffe2.sh](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/tutorials/dlcookbook/caffe2.sh), [nvidia_caffe.sh](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/tutorials/dlcookbook/nvidia_caffe.sh), [tensorflow.sh](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/tutorials/dlcookbook/tensorflow.sh), [tensorrt.sh](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/tutorials/dlcookbook/tensorrt.sh) demonstrate minimal working examples how to run benchmarks with these frameworks. In order to run them, you need to have these frameworks installed in a host OS or as a docker container.
3. [log_parser.sh](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/tutorials/dlcookbook/log_parser.sh) demonstrates how it's possible to parse benchmarks' logs files.
4. [summary_builder.sh](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/tutorials/dlcookbook/summary_builder.sh) demonstrates how to build simple reports.
