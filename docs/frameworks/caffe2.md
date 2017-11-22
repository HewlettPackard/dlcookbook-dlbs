
# __Caffe2__

We've integrated caffe2 into benchmarking suite using similar to TensorFlow approach. We've written [caffe2_benchmarks](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/caffe2_benchmarks) python projects that exposes similar command line API as tf_cnn_benchmarks. A Caffe2 launcher shell [script](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/scripts/launchers/caffe2.sh) serves as mediator between experimenter and caffe2 benchmarks project.

## Standalone run
The project itself is located in [caffe2_benchmarks](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/caffe2_benchmarks) folder. The [benchmarks.py](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/benchmarks.py) file is the entry point. Following command line parameters are supported:
1.  `--model` (type=str) A model to benchmark ("alexnet", "googlenet" ...).
2.  `--forward_only` (type=str) Benchmark inference (if true) else benchmark training.
3.  `--batch_size` (type=int) Per device batch size.
4.  `--num_batches` (type=int) Number of benchmark iterations.
5.  `--num_warmup_batches` (type=int) Number of warmup iterations.
6.  `--num_gpus` (type=int) Number of gpus to use (per node?). Use CUDA_VISIBLE_DEVICES to select those devices.
7.  `--device` (type=str) Comptue device, "cpu" or "gpu".
8.  `--data_dir` (type=str) Path to the LMDB or LEVELDB data base.
9.  `--data_backend` (choices=['lmdb', 'leveldb']) One of "lmdb" or "leveldb".
10. `--dtype` (choices=['float', 'float32', 'float16']) Precision of data variables: float(same as float32), float32 or float16.
11. `--enable_tensor_core` Enable volta's tensor ops (requires CUDA >= 9, cuDNN >= 7 and NVIDIA Volta GPU).

## Models
The class `ModelFactory` in [model_factory.py](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/model_factory.py) is responsible for creating neural networks.
Complete list of supported models can be found on the [model](/models/models.md?id=supported-models) page.

## Adding new model
To add new model, several steps need to be performed:
1. Add a new file in [models](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/caffe2_benchmarks/models) folder. Create a class and inherit it from [caffe2_benchmarks.models.model.Model](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/models/model.py).
2. Study any implementation of the existing model. The [DeepMNIST](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/models/deep_mnist.py) model is probably the simplest one.
    1. The constructor of a model class accepts one parameter - `params` - a dictionary that defines model property. In particular, it will contain at least these parameters: `model` (model id), `phase` (`training` or `inference`), `batch_size` (bath size) and `dtype` (type of data to use, for instance, float). You then must ensure this dictionary also contains `name` (model name), `input_shape` (shape of input tensor excluding batch dimension) and `num_classes` (number of output classes). Base class then is initialized with parameters.
    2. Implement the `forward_pass_builder` method (see existing implementations in `models` folder for examples).

## Commonly used configuration parameters

### __caffe2.docker.image__ = `hpe/caffe2:cuda9-cudnn7`
The name of a docker image for Caffe2 if containerized benchmark is requested.
### __caffe2.host.python_path__ = `${HOME}/projects/caffe2/build`
Path to a Caffe2's python folder in case of a bare metal run.
### __caffe2.host.libpath__ = `${HOME}/projects/caffe2/build/caffe2`
Basically, it's a LD_LIBRARY_PATH for MXNet in case of a bare metal run.

## Other parameters

### __caffe2.data_dir__ = `""`
A data directory if real data should be used. If empty, synthetic data is used
(no data ingestion pipeline).
### __caffe2.data_backend__ = `lmdb`
In case of real data, specifies its storage backend ('lmdb').
### __caffe2.dtype__ = `${exp.dtype}`
Precision to use - `float32`(`float`) or `float16`.
### __caffe2.enable_tensor_core__ = `${exp.enable_tensor_core}`
Enable/disable tensor core operations (only for VOLTA generation NVIDIA GPUs
with CUDA 9).

## Internal parameters

### __caffe2.launcher__ = `${DLBS_ROOT}/scripts/launchers/caffe2.sh`
Path to script that launches Caffe2 benchmarks.
### __caffe2.args__ = ..
Command line arguments that launcher uses to launch caffe2_benchmark script.
### __caffe2.docker.args__ = ..
In case if containerized benchmarks, this are the docker parameters for Caffe2.
### __caffe2.bench_path__ = `$('${DLBS_ROOT}/python' if '${exp.env}' == 'host' else '/workspace')$`
Python path to where mxnet_benchmarks project is located. Depends on bare
metal/docker benchmark.
